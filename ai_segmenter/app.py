import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import customtkinter as ctk
from PIL import Image
import threading
import sys
import urllib.request
import ssl
import subprocess
import time
import logging
import warnings
from collections import deque

from ai_segmenter.app_icons import apply_window_icon
from ai_segmenter.config import (
    CORRIDORKEY_DEVICE_OPTIONS,
    DECKLINK_OUTPUT_MODES,
    MAIN_AI_DEVICE_OPTIONS,
    MODEL_OPTIONS,
    YOLO_DEVICE_OPTIONS,
    YOLO_MODEL_OPTIONS,
)
from ai_segmenter.decklink import get_decklink_output_devices
from ai_segmenter.image_utils import (
    center_crop_resize_rgb as _center_crop_resize_rgb,
)
from ai_segmenter.live_output import LiveOutputMixin
from ai_segmenter.metrics import MetricsMixin
from ai_segmenter.models import CorridorKeyRefiner, YoloObjectDetector, create_segmentation_model
from ai_segmenter.pipeline import CameraLifecycleMixin, LivePipelineMixin
from ai_segmenter.postprocessing import PostProcessingMixin
from ai_segmenter.preview_renderer import PreviewRendererMixin
from ai_segmenter.ui import AppLayoutMixin
from ai_segmenter.utils import run_with_timeout
from ai_segmenter.yolo_controls import YoloControlsMixin

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")

class FoolproofSyncApp(
    MetricsMixin,
    LiveOutputMixin,
    PostProcessingMixin,
    PreviewRendererMixin,
    YoloControlsMixin,
    AppLayoutMixin,
    LivePipelineMixin,
    CameraLifecycleMixin,
):
    def __init__(self, root):
        self.root = root
        self.root.title("AI Segmenter - Multi Model")
        apply_window_icon(self.root)
        self.root.geometry("1250x850")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.model_path = "selfie_multiclass.tflite"
        self.check_and_download_model()

        self.ui_w, self.ui_h = 800, 450
        self.preview_display_w, self.preview_display_h = self.ui_w, self.ui_h
        self.control_panel_width = 340
        self.control_panel_min_width = 260
        self.control_panel_max_width = 560
        self._control_resize_start_x = 0
        self._control_resize_start_width = self.control_panel_width
        self.metrics_expanded = ctk.BooleanVar(value=True)

        self.model_name = ctk.StringVar(value=MODEL_OPTIONS[0])
        self.main_ai_device_mode = ctk.StringVar(value=MAIN_AI_DEVICE_OPTIONS[0])
        self.model_lock = threading.Lock()
        self.segmenter = create_segmentation_model(
            self.model_name.get(),
            self.model_path,
            self._resolve_main_ai_force_device()
        )
        self.loaded_model_name = self.model_name.get()
        self.model_status = self._format_model_status(self.segmenter, self.loaded_model_name)

        self.cap = None
        self.is_running = False
        self.pipeline_stop_event = threading.Event()
        self.pipeline_threads = []
        self.ai_frame_condition = threading.Condition()
        self.ai_latest_frame = None
        self.ai_latest_frame_id = 0
        self.ai_last_taken_frame_id = 0
        self.output_condition = threading.Condition()
        self.output_latest_payload = None
        self.output_latest_frame_id = 0
        self.output_last_taken_frame_id = 0
        self.pipeline_bg_cache = None
        self.pipeline_gpu_bg_cache = None
        self.pipeline_last_bg_mode = ""
        self.pipeline_last_capture_done_time = 0.0
        self.current_camera_index = 0
        self.current_live_source = None

        self.bg_mode = ctk.StringVar(value="Checker")
        self.view_mode = ctk.StringVar(value="Processed")
        self.custom_background_image = None
        self.custom_background_source = None
        self.checker_background_source = None
        self.force_bg_update = False
        self.yolo_enabled = ctk.BooleanVar(value=False)
        self.yolo_model_name = ctk.StringVar(value=YOLO_MODEL_OPTIONS[0])
        self.yolo_device_mode = ctk.StringVar(value=YOLO_DEVICE_OPTIONS[0])
        self.yolo_confidence = ctk.DoubleVar(value=0.25)
        self.yolo_detector = None
        self.yolo_post_detector = None
        self.yolo_model_lock = threading.Lock()
        self.yolo_status = ctk.StringVar(value="YOLO aus / als Modell waehlbar")
        self.yolo_detections = []
        self.yolo_selected_keys = set()
        self.yolo_seen_keys = set()
        self.yolo_detection_vars = {}
        self.yolo_frame_counter = 0
        self.yolo_detect_every = 3
        self.yolo_last_detect_time = 0.0
        self.yolo_min_detect_interval = 0.12
        self.yolo_last_detection_signature = None
        self.yolo_last_ui_update_time = 0.0
        self.yolo_ui_update_interval = 0.2
        self.yolo_last_analysis_ms = 0.0
        self.yolo_last_analysis_time = 0.0
        self.yolo_async_lock = threading.Lock()
        self.yolo_async_running = False
        self.yolo_async_pending_frame = None
        self.yolo_async_stop = False
        self.yolo_async_thread = None
        self.yolo_select_all = ctk.BooleanVar(value=True)
        self.yolo_sync_postprocess = ctk.BooleanVar(value=False)
        self.corridor_enabled = ctk.BooleanVar(value=False)
        self.corridor_device_mode = ctk.StringVar(value=CORRIDORKEY_DEVICE_OPTIONS[0])
        self.corridor_despill_strength = ctk.DoubleVar(value=0.45)
        self.corridor_despeckle_size = ctk.IntVar(value=400)
        self.corridor_refiner = None
        self.corridor_lock = threading.Lock()
        self.corridor_status = ctk.StringVar(value="CorridorKey aus")

        self.edge_erode = ctk.IntVar(value=3)
        self.edge_soft = ctk.IntVar(value=7)
        self.fast_live_alpha = ctk.BooleanVar(value=True)

        self.latest_pil_image = None
        self.latest_display_payload = None
        self.latest_display_payload_id = 0
        self.displayed_payload_id = -1
        self.rendered_display_payload_id = -1
        self.rendered_display_size = (0, 0)
        self.empty_dummy_image = ctk.CTkImage(light_image=Image.new("RGB", (1, 1)), size=(1, 1))

        self.metrics_lock = threading.Lock()
        self.metrics_text = "Performance\nKamera gestoppt"
        self.metrics_last_gui_text = None
        self.pipeline_profiler = None
        self.pipeline_log_path = None
        self._reset_perf_metrics()
        self.app_mode = ctk.StringVar(value="Live")
        self.post_input_path = ctk.StringVar(value="Keine Datei gewaehlt")
        self.post_output_path = ctk.StringVar(value="Kein Ziel gewaehlt")
        self.post_status = ctk.StringVar(value="Bereit")
        self.post_is_processing = False
        decklink_devices = run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=5.0)
        self.live_output_enabled = ctk.BooleanVar(value=False)
        self.live_output_device = ctk.StringVar(value=decklink_devices[0])
        self.live_key_output_enabled = ctk.BooleanVar(value=False)
        key_device_default = decklink_devices[1] if len(decklink_devices) > 1 else decklink_devices[0]
        self.live_key_output_device = ctk.StringVar(value=key_device_default)
        self.live_output_mode = ctk.StringVar(value=next(iter(DECKLINK_OUTPUT_MODES)))
        self.live_output_status = ctk.StringVar(value="DeckLink Output aus")
        self.decklink_output = None
        self.decklink_key_output = None
        self.sync_overlay_mode = ctk.StringVar(value="Aus")
        self.fill_delay_frames = ctk.IntVar(value=0)
        self.matte_delay_frames = ctk.IntVar(value=0)
        self.live_output_frame_counter = 0
        self.fill_delay_buffer = deque()
        self.matte_delay_buffer = deque()

        self.setup_gui()
        self.root.after(250, self.refresh_cameras)
        self.update_gui_loop()

    def _resolve_main_ai_force_device(self):
        mode = self.main_ai_device_mode.get()
        if mode == "CPU":
            return "cpu"
        if mode == "CUDA":
            return "cuda"
        return None

    def _main_ai_device_mode_note(self):
        return self.main_ai_device_mode.get()

    def _format_model_status(self, segmenter, choice):
        if choice in ("YOLO", "YOLO TensorRT"):
            with self.yolo_model_lock:
                detector = self.yolo_detector
            if detector is not None:
                base = f"Modell bereit: {choice} ({detector.backend_label})"
                if detector.device_hint:
                    return base + "\n" + str(detector.device_hint)
                return base
            return f"{choice} wird ueber die Objektauswahl geladen"
        device_label = getattr(segmenter, "device_label", None)
        device_hint = getattr(segmenter, "device_hint", None)
        if device_label:
            base = f"Modell bereit: {choice} ({device_label})"
        else:
            base = f"Modell bereit: {choice}"
        if device_hint:
            return base + "\n" + str(device_hint)
        return base

    def check_and_download_model(self):
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 100000:
            print("Lade KI-Modell...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            try:
                with urllib.request.urlopen(url, context=ctx) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Fehler beim Modell-Download: {e}")
                sys.exit(1)



    def toggle_corridor_key(self):
        if not self.corridor_enabled.get():
            with self.corridor_lock:
                self.corridor_refiner = None
            self.corridor_status.set("CorridorKey aus")
            return
        self.corridor_status.set(f"CorridorKey wird geladen ({self.corridor_device_mode.get()}) ...")
        threading.Thread(target=self._load_corridor_worker, daemon=True).start()

    def change_corridor_device(self, choice):
        with self.corridor_lock:
            self.corridor_refiner = None
        if self.corridor_enabled.get():
            self.corridor_status.set(f"CorridorKey wird auf {choice} geladen ...")
            threading.Thread(target=self._load_corridor_worker, daemon=True).start()
        else:
            self.corridor_status.set(f"CorridorKey aus / Hardware: {choice}")

    def _update_corridor_status_settings(self):
        if not self.corridor_enabled.get():
            self.corridor_status.set(
                f"CorridorKey aus / Despill {self.corridor_despill_strength.get():.2f} | "
                f"Despeckle {int(self.corridor_despeckle_size.get())}"
            )
            return
        with self.corridor_lock:
            refiner = self.corridor_refiner
        if refiner is None:
            return
        self.corridor_status.set(
            f"CorridorKey bereit ({refiner.device_label}, {refiner.img_size}px)\n"
            f"Despill {self.corridor_despill_strength.get():.2f} | "
            f"Despeckle {int(self.corridor_despeckle_size.get())}"
        )

    def _load_corridor_worker(self):
        try:
            refiner = CorridorKeyRefiner(device_mode=self.corridor_device_mode.get())
            with self.corridor_lock:
                self.corridor_refiner = refiner
            self.root.after(
                0,
                lambda: self.corridor_status.set(
                    f"CorridorKey bereit ({refiner.device_label}, {refiner.img_size}px)\n"
                    f"Despill {self.corridor_despill_strength.get():.2f} | "
                    f"Despeckle {int(self.corridor_despeckle_size.get())}"
                )
            )
        except Exception as exc:
            with self.corridor_lock:
                self.corridor_refiner = None
            self.root.after(0, lambda e=exc: self.corridor_status.set(f"CorridorKey Fehler: {e}"))

    def _apply_corridor_key(self, rgb_frame, alpha_2d):
        if not self.corridor_enabled.get():
            return rgb_frame, alpha_2d
        with self.corridor_lock:
            refiner = self.corridor_refiner
        if refiner is None:
            return rgb_frame, alpha_2d
        try:
            return refiner.refine(
                rgb_frame,
                alpha_2d,
                despill_strength=self.corridor_despill_strength.get(),
                despeckle_size=self.corridor_despeckle_size.get(),
            )
        except Exception as exc:
            with self.corridor_lock:
                self.corridor_refiner = None
            self.root.after(0, lambda e=exc: self.corridor_status.set(f"CorridorKey Fehler: {e}"))
            return rgb_frame, alpha_2d


    def change_app_mode(self, mode):
        if mode == "Postproduktion":
            if self.is_running:
                self._stop_camera_internal()
            self.stop_live_output()
            self.live_frame.pack_forget()
            self.post_frame.pack(pady=(0, 8), padx=0, fill="x")
            self.video_label.configure(image=self.empty_dummy_image, text="Postproduktion bereit")
            with self.metrics_lock:
                if not self.post_is_processing:
                    self.metrics_text = "Performance\nPostproduktion bereit"
            return

        self.post_frame.pack_forget()
        self.live_frame.pack(pady=(0, 8), padx=0, fill="x")
        if not self.is_running:
            self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")
            self._reset_perf_metrics()


    def change_main_ai_device(self, choice):
        if self._is_yolo_primary_model():
            self.model_status = "YOLO nutzt die separate YOLO-Hardwareauswahl."
            self.model_status_label.configure(text=self.model_status)
            return
        self.change_model(self.loaded_model_name)

    def change_model(self, choice):
        was_running = self.is_running
        if was_running:
            self._stop_camera_internal()

        self.model_select.configure(state="disabled")
        if hasattr(self, "main_ai_device_select"):
            self.main_ai_device_select.configure(state="disabled")
        self.video_label.configure(image=self.empty_dummy_image, text=f"Lade Modell: {choice}...")
        self.model_status_label.configure(text=f"Lade Modell: {choice} ({self._main_ai_device_mode_note()})...")
        threading.Thread(target=self._load_model_worker, args=(choice, was_running), daemon=True).start()

    def _load_model_worker(self, choice, restart_camera):
        try:
            if choice in ("YOLO", "YOLO TensorRT"):
                use_yolo_trt = choice == "YOLO TensorRT"
                detector = YoloObjectDetector(
                    self.yolo_model_name.get(),
                    force_device=self._resolve_yolo_force_device(False),
                    prefer_tensorrt=use_yolo_trt,
                    build_tensorrt_engine=use_yolo_trt,
                )
                with self.yolo_model_lock:
                    self.yolo_detector = detector
                with self.model_lock:
                    self.segmenter = None
                status = f"Modell bereit: {choice} ({detector.backend_label})"
                if detector.device_hint:
                    status += "\n" + str(detector.device_hint)
                self.model_status = status
                self.root.after(
                    0,
                    lambda: self.yolo_status.set(
                        f"{choice} als Hauptmodell aktiv: {self.yolo_model_name.get()} ({detector.backend_label})"
                    )
                )
            else:
                new_segmenter = create_segmentation_model(
                    choice,
                    self.model_path,
                    self._resolve_main_ai_force_device()
                )
                with self.model_lock:
                    self.segmenter = new_segmenter
                self.model_status = self._format_model_status(new_segmenter, choice)
            self.root.after(0, lambda: self._finish_model_load(choice, restart_camera, None))
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._finish_model_load(choice, False, error))

    def _finish_model_load(self, choice, restart_camera, error):
        self.model_select.configure(state="normal")
        if hasattr(self, "main_ai_device_select"):
            self.main_ai_device_select.configure(state="normal")
        if error is None:
            self.model_name.set(choice)
            self.loaded_model_name = choice
            self.model_status_label.configure(text=self.model_status)
            if choice not in ("YOLO", "YOLO TensorRT") and not self.yolo_enabled.get():
                self.yolo_detections = []
                self.yolo_selected_keys.clear()
                self.yolo_seen_keys.clear()
                self.yolo_status.set("YOLO-Nachbearbeitung aus")
                self._refresh_yolo_detection_list()
            if self.app_mode.get() == "Postproduktion":
                self.video_label.configure(image=self.empty_dummy_image, text="Postproduktion bereit")
            else:
                self.video_label.configure(image=self.empty_dummy_image, text="Kamera gestoppt")
            if restart_camera:
                self.root.after(300, self._start_camera_internal)
            return

        self.model_name.set(self.loaded_model_name)
        self.model_status = f"Fehler beim Laden von {choice}: {error}"
        self.model_status_label.configure(text=self.model_status)
        self.video_label.configure(image=self.empty_dummy_image, text=self.model_status)

    def trigger_background_load(self):
        threading.Thread(target=self._applescript_worker, daemon=True).start()

    def _applescript_worker(self):
        # Automatische Erkennung ob Mac oder Windows!
        if sys.platform == "darwin":
            script = '''
            set f to choose file with prompt "Hintergrundbild wählen"
            POSIX path of f
            '''
            try:
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                if result.returncode == 0:
                    file_path = result.stdout.strip()
                    self._load_image_from_path(file_path)
            except Exception as e:
                print(f"macOS Finder Fehler: {e}")
        else:
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.jpg;*.jpeg;*.png")])
            if file_path:
                self._load_image_from_path(file_path)

    def _load_image_from_path(self, file_path):
        if file_path and os.path.exists(file_path):
            try:
                pil_img = Image.open(file_path).convert('RGB')
                bg_rgb = np.array(pil_img)
                self.custom_background_source = bg_rgb
                self.custom_background_image = _center_crop_resize_rgb(bg_rgb, self.ui_w, self.ui_h)
                self.force_bg_update = True
                self.root.after(0, lambda: self.bg_mode.set("CustomImage"))
            except Exception as img_e:
                print(f"Fehler beim Verarbeiten des Bildes: {img_e}")

