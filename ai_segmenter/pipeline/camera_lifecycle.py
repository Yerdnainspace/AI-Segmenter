import threading
import time

import cv2

from ai_segmenter.camera import get_available_cameras, measure_camera_input_fps, open_camera, parse_camera_index
from ai_segmenter.decklink import DeckLinkLiveInput, get_decklink_input_devices
from ai_segmenter.utils import run_with_timeout


class CameraLifecycleMixin:
    def refresh_cameras(self):
        if getattr(self, "_camera_refresh_running", False):
            return
        self._camera_refresh_running = True
        try:
            self.btn_refresh_cameras.configure(text="Suche laeuft...", state="disabled")
        except Exception:
            pass

        def worker():
            try:
                decklink_inputs = run_with_timeout(get_decklink_input_devices, [], timeout=5.0)
                decklink_sources = [f"DeckLink: {name}" for name in decklink_inputs]
                cameras = run_with_timeout(get_available_cameras, ["Keine Kamera gefunden"], timeout=8.0)
                if cameras == ["Keine Kamera gefunden"]:
                    cameras = []
                values = [*decklink_sources, *cameras]
                if not values:
                    values = ["Keine Live-Quelle gefunden"]
                error = None
            except Exception as exc:
                values = None
                error = exc
            self.root.after(0, lambda: self._finish_camera_refresh(values, error))

        threading.Thread(target=worker, name="CameraRefresh", daemon=True).start()

    def _finish_camera_refresh(self, values, error=None):
        self._camera_refresh_running = False
        try:
            self.btn_refresh_cameras.configure(text="Kameras neu suchen", state="normal")
        except Exception:
            pass
        if error is not None:
            self.video_label.configure(text=f"Kamerasuche Fehler: {error}")
            return
        if not values:
            values = ["Keine Live-Quelle gefunden"]

        previous_source = self.current_live_source or self.camera_select.get()
        self.camera_select.configure(values=values)
        selected_source = previous_source if previous_source in values else values[0]
        self.camera_select.set(selected_source)
        self.current_live_source = selected_source
        if selected_source != "Keine Live-Quelle gefunden" and not selected_source.startswith("DeckLink: "):
            try:
                self.current_camera_index = parse_camera_index(selected_source)
            except Exception:
                self.current_camera_index = 0

    def change_camera(self, choice):
        try:
            if choice == self.current_live_source:
                return

            was_running = self.is_running
            if was_running:
                self._stop_camera_internal(preserve_preview=True)

            self.current_live_source = choice
            if choice != "Keine Live-Quelle gefunden" and not choice.startswith("DeckLink: "):
                self.current_camera_index = parse_camera_index(choice)

            if was_running:
                self.root.after(150, lambda: self._start_camera_internal(preserve_preview=True))
        except Exception as e:
            print(f"Fehler beim Kamerawechsel: {e}")

    def toggle_camera(self):
        if not self.is_running:
            self._start_camera_internal()
        else:
            self._stop_camera_internal()

    def _start_camera_internal(self, preserve_preview=False):
        if self.is_running: return

        self.yolo_async_stop = False
        if not preserve_preview:
            self.latest_pil_image = None
            self.latest_display_payload = None
            self.latest_display_payload_id = 0
            self.displayed_payload_id = -1
            self.rendered_display_payload_id = -1
            self.rendered_display_size = (0, 0)
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera startet...")

        source = self.current_live_source or self.camera_select.get()
        if source == "Keine Live-Quelle gefunden":
            self.video_label.configure(image=self.empty_dummy_image, text="Keine Live-Quelle gefunden")
            return
        if source.startswith("DeckLink: "):
            device_name = source.split("DeckLink: ", 1)[1]
            self.cap = DeckLinkLiveInput(device_name, self.live_output_mode.get())
            try:
                self.cap.open()
            except Exception as exc:
                self.cap = None
                self.video_label.configure(text=f"DeckLink Input Fehler: {exc}")
                return
            backend_name = "DeckLink SDK"
        else:
            self.cap, backend_name = open_camera(self.current_camera_index)

        if self.cap is not None and self.cap.isOpened():
            if not isinstance(self.cap, DeckLinkLiveInput):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            measured_fps = 0.0 if isinstance(self.cap, DeckLinkLiveInput) else measure_camera_input_fps(self.cap)
            if measured_fps > 1.0:
                actual_fps = measured_fps
            elif reported_fps > 1.0 and reported_fps <= 240.0:
                actual_fps = reported_fps
            else:
                actual_fps = 0.0
            self._reset_perf_metrics(source_fps=actual_fps, backend_name=backend_name, capture_size=(actual_w, actual_h))
            self._start_pipeline_profiler(
                source=source,
                backend_name=backend_name,
                capture_size=(actual_w, actual_h),
                source_fps=actual_fps,
            )

            self.is_running = True
            self.btn_toggle.configure(text="Kamera Stoppen", fg_color="#d32f2f", hover_color="#9a0007")
            self.video_label.configure(text="")

            self._start_live_pipeline_threads()
        else:
            self.video_label.configure(text="Fehler: Kamera besetzt oder nicht verfuegbar")

    def _stop_camera_internal(self, preserve_preview=False):
        self.is_running = False
        self.pipeline_stop_event.set()
        with self.ai_frame_condition:
            self.ai_frame_condition.notify_all()
        with self.output_condition:
            self.output_condition.notify_all()
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self._stop_pipeline_profiler()
            self.root.after(0, lambda: self.btn_toggle.configure(text="Kamera Starten", fg_color=["#3a7ebf", "#1f538d"]))
        with self.output_condition:
            self.output_condition.notify_all()
        for thread in list(self.pipeline_threads):
            if thread.is_alive():
                thread.join(timeout=1.5)
        self.pipeline_threads = []
        self.ai_latest_frame = None
        self.output_latest_payload = None
        with self.yolo_async_lock:
            self.yolo_async_pending_frame = None
        self._stop_pipeline_profiler()
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_toggle.configure(text="Kamera Starten", fg_color=["#3a7ebf", "#1f538d"])
        if not preserve_preview:
            self.latest_pil_image = None
            self.latest_display_payload = None
            self.latest_display_payload_id = 0
            self.displayed_payload_id = -1
            self.rendered_display_payload_id = -1
            self.rendered_display_size = (0, 0)
        self._reset_perf_metrics()
        if not preserve_preview:
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")

    def on_closing(self):
        self.yolo_async_stop = True
        with self.yolo_async_lock:
            self.yolo_async_pending_frame = None
        self._stop_camera_internal()
        self.stop_live_output()
        self.root.destroy()
