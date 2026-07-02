import threading
import time

import cv2
import customtkinter as ctk

from ai_segmenter.decklink import DeckLinkLiveInput
from ai_segmenter.runtime import TENSORRT_RUNTIME_LOCK


class LivePipelineMixin:
    def _start_live_pipeline_threads(self):
        self.pipeline_stop_event.clear()
        self.ai_latest_frame = None
        self.ai_latest_frame_id = 0
        self.ai_last_taken_frame_id = 0
        self.output_latest_payload = None
        self.output_latest_frame_id = 0
        self.output_last_taken_frame_id = 0
        self.pipeline_last_capture_done_time = 0.0
        self.pipeline_bg_cache = None
        self.pipeline_gpu_bg_cache = None
        self.pipeline_last_bg_mode = ""
        self.pipeline_threads = [
            threading.Thread(target=self.video_capture_loop, name="LiveCapture", daemon=True),
            threading.Thread(target=self.video_ai_loop, name="LiveAI", daemon=True),
            threading.Thread(target=self.video_output_loop, name="LiveDeckLinkOutput", daemon=True),
        ]
        for thread in self.pipeline_threads:
            thread.start()

    def update_gui_loop(self):
        if (
            (self.is_running or self.app_mode.get() == "Postproduktion")
            and self.latest_display_payload is not None
            and self.displayed_payload_id != self.latest_display_payload_id
        ):
            rgb_frame, alpha_2d, processed_frame = self.latest_display_payload
            self.latest_pil_image = self._make_display_image(rgb_frame, alpha_2d, processed_frame)
            self.displayed_payload_id = self.latest_display_payload_id
        if (self.is_running or self.app_mode.get() == "Postproduktion") and self.latest_pil_image is not None:
            render_size = (int(self.preview_display_w), int(self.preview_display_h))
            if (
                self.rendered_display_payload_id != self.displayed_payload_id
                or self.rendered_display_size != render_size
            ):
                img_tk = ctk.CTkImage(
                    light_image=self.latest_pil_image,
                    size=render_size,
                )
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk
                self.rendered_display_payload_id = self.displayed_payload_id
                self.rendered_display_size = render_size
        if self.is_running and self.pipeline_profiler is not None:
            pipeline_text = self._format_pipeline_summary(self.pipeline_profiler.get_last_summary())
            if pipeline_text:
                with self.metrics_lock:
                    self.metrics_text = pipeline_text
        with self.metrics_lock:
            metrics_text = self.metrics_text
        if metrics_text != self.metrics_last_gui_text:
            if hasattr(self, "metrics_summary_label"):
                self.metrics_summary_label.configure(text=self._metrics_summary_text(metrics_text))
            self.metrics_label.configure(text=metrics_text)
            self.metrics_last_gui_text = metrics_text
        self.root.after(33, self.update_gui_loop)

    def _read_capture_frame_timed(self):
        if isinstance(self.cap, DeckLinkLiveInput):
            wait_start = time.perf_counter()
            ret, frame, capture_actual_s = self.cap.read_with_timing()
            wait_done = time.perf_counter()
            wait_s = max(0.0, (wait_done - wait_start) - capture_actual_s)
            return ret, frame, wait_s, capture_actual_s

        grab_start = time.perf_counter()
        try:
            grabbed = self.cap.grab()
        except Exception:
            grabbed = False
        grab_done = time.perf_counter()
        grab_s = grab_done - grab_start
        if not grabbed:
            return False, None, grab_s, 0.0

        retrieve_start = time.perf_counter()
        ret, frame = self.cap.retrieve()
        retrieve_done = time.perf_counter()
        retrieve_s = retrieve_done - retrieve_start

        if ret:
            # Depending on the OpenCV backend either grab() or retrieve() can block
            # until the next frame arrives. The shorter part is the practical
            # handoff/decode cost; the longer blocking part belongs to frame wait.
            if grab_s > 0.0 and retrieve_s > 0.0:
                capture_actual_s = min(grab_s, retrieve_s)
            else:
                capture_actual_s = max(grab_s, retrieve_s)
            wait_s = max(0.0, (retrieve_done - grab_start) - capture_actual_s)
        else:
            capture_actual_s = 0.0
            wait_s = retrieve_done - grab_start
        return ret, frame, wait_s, capture_actual_s

    def video_capture_loop(self):
        while self.is_running and not self.pipeline_stop_event.is_set():
            if self.cap is None:
                break
            profiler = self.pipeline_profiler
            ret, frame, capture_wait_s, capture_actual_s = self._read_capture_frame_timed()
            read_done = time.perf_counter()
            if profiler is not None:
                profiler.sample("capture_read", capture_wait_s)
                profiler.sample("capture_actual", capture_actual_s)
                if ret and self.pipeline_last_capture_done_time > 0:
                    profiler.sample("capture_interval", read_done - self.pipeline_last_capture_done_time)
                if isinstance(self.cap, DeckLinkLiveInput):
                    received_delta, overwritten_delta = self.cap.consume_frame_stats()
                    profiler.count("source_arrival", received_delta)
                    profiler.count("source_overwrite", overwritten_delta)
            if not ret:
                self._perf_read_failures += 1
                if profiler is not None:
                    profiler.count("read_fail")
                    profiler.flush_if_due()
                if isinstance(self.cap, DeckLinkLiveInput):
                    time.sleep(0.005)
                    continue
                break

            self.pipeline_last_capture_done_time = read_done
            if profiler is not None:
                profiler.count("capture")
            preprocess_start = time.perf_counter()
            ui_frame = cv2.resize(frame, (self.ui_w, self.ui_h))
            rgb_frame = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)
            if profiler is not None:
                profiler.sample("preprocess", time.perf_counter() - preprocess_start)
                profiler.count("preprocess")
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

            with self.ai_frame_condition:
                if self.ai_latest_frame is not None and self.ai_latest_frame_id != self.ai_last_taken_frame_id:
                    if profiler is not None:
                        profiler.count("ai_input_overwrite")
                self.ai_latest_frame_id += 1
                self.ai_latest_frame = (self.ai_latest_frame_id, rgb_frame, frame.shape, read_done, preprocess_ms)
                self.ai_frame_condition.notify()

        self.pipeline_stop_event.set()
        with self.ai_frame_condition:
            self.ai_frame_condition.notify_all()

    def _take_latest_ai_frame(self):
        with self.ai_frame_condition:
            while (
                self.is_running
                and not self.pipeline_stop_event.is_set()
                and (self.ai_latest_frame is None or self.ai_latest_frame_id == self.ai_last_taken_frame_id)
            ):
                self.ai_frame_condition.wait(0.1)
            if self.ai_latest_frame is None or self.ai_latest_frame_id == self.ai_last_taken_frame_id:
                return None
            self.ai_last_taken_frame_id = self.ai_latest_frame_id
            return self.ai_latest_frame

    def _queue_output_frame(self, frame_id, rgb_frame, alpha_2d, output_final):
        with self.output_condition:
            if self.output_latest_payload is not None and self.output_latest_frame_id != self.output_last_taken_frame_id:
                profiler = self.pipeline_profiler
                if profiler is not None:
                    profiler.count("output_overwrite")
            self.output_latest_frame_id = frame_id
            self.output_latest_payload = (frame_id, rgb_frame, alpha_2d, output_final)
            self.output_condition.notify()

    def video_ai_loop(self):
        while self.is_running and not self.pipeline_stop_event.is_set():
            item = self._take_latest_ai_frame()
            if item is None:
                continue
            frame_id, rgb_frame, original_shape, frame_start, preprocess_ms = item
            profiler = self.pipeline_profiler
            gpu_alpha_t = None
            gpu_segmenter = None
            fast_alpha_lock_held = False
            try:
                infer_start = time.perf_counter()
                if self._is_yolo_primary_model():
                    self._run_yolo_primary(rgb_frame)
                    mask_binary = None
                    alpha_2d = self._make_yolo_alpha(rgb_frame.shape)
                else:
                    with self.model_lock:
                        segmenter = self.segmenter
                    if (
                        self._can_use_fast_live_alpha(segmenter)
                        and hasattr(segmenter, "predict_alpha_tensor")
                        and getattr(segmenter, "device_label", "") == "CUDA"
                    ):
                        if getattr(segmenter, "tensorrt_enabled", False):
                            TENSORRT_RUNTIME_LOCK.acquire()
                            fast_alpha_lock_held = True
                        gpu_alpha_t = segmenter.predict_alpha_tensor(rgb_frame)
                        gpu_segmenter = segmenter
                        mask_binary = None
                    else:
                        mask_binary = segmenter.predict_mask(rgb_frame)
                    current_status = self._format_model_status(segmenter, self.loaded_model_name)
                    if current_status != self.model_status:
                        self.model_status = current_status
                        self.root.after(0, lambda text=current_status: self.model_status_label.configure(text=text))
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                if profiler is not None:
                    profiler.sample("ai", infer_ms / 1000.0)
                    profiler.count("ai")
            except Exception as model_error:
                if fast_alpha_lock_held:
                    TENSORRT_RUNTIME_LOCK.release()
                    fast_alpha_lock_held = False
                self.is_running = False
                self.pipeline_stop_event.set()
                self.root.after(0, lambda e=model_error: self.video_label.configure(
                    image=self.empty_dummy_image,
                    text=f"Modellfehler: {e}"
                ))
                break

            post_start = time.perf_counter()
            used_gpu_compose = False
            compose_rgb_frame = rgb_frame
            if gpu_alpha_t is not None and gpu_segmenter is not None:
                alpha_start = time.perf_counter()
                try:
                    output_final, alpha_2d = self._gpu_postprocess_compose(gpu_segmenter, rgb_frame, gpu_alpha_t)
                    used_gpu_compose = True
                except Exception as fast_alpha_error:
                    self.fast_live_alpha.set(False)
                    status = (
                        "Live Fast Alpha deaktiviert: GPU-Pfad ist fehlgeschlagen; "
                        f"normaler Alpha-Pfad wird genutzt. Originalfehler: {fast_alpha_error}"
                    )
                    self.root.after(0, lambda text=status: self.model_status_label.configure(text=text))
                    mask_binary = gpu_segmenter.predict_mask(rgb_frame)
                    alpha_2d = self._postprocess_to_alpha(rgb_frame, mask_binary)
                finally:
                    if fast_alpha_lock_held:
                        TENSORRT_RUNTIME_LOCK.release()
                        fast_alpha_lock_held = False
                if profiler is not None:
                    profiler.sample("alpha_post", time.perf_counter() - alpha_start)
                    profiler.count("alpha_post")
            elif not self._is_yolo_primary_model():
                alpha_start = time.perf_counter()
                alpha_2d = self._postprocess_to_alpha(rgb_frame, mask_binary)
                if profiler is not None:
                    profiler.sample("alpha_post", time.perf_counter() - alpha_start)
                    profiler.count("alpha_post")
                if self.yolo_sync_postprocess.get():
                    yolo_post_start = time.perf_counter()
                    self._run_yolo_postprocess_sync(rgb_frame)
                    if profiler is not None:
                        profiler.sample("yolo_post", time.perf_counter() - yolo_post_start)
                        profiler.count("yolo_post")
                else:
                    self._schedule_yolo_postprocess(rgb_frame)
                alpha_2d = self._apply_yolo_roi_to_alpha(alpha_2d)

            if used_gpu_compose:
                corridor_start = time.perf_counter()
                if profiler is not None:
                    profiler.sample("corridor", time.perf_counter() - corridor_start)
                    profiler.count("corridor")
            else:
                corridor_start = time.perf_counter()
                compose_rgb_frame, alpha_2d = self._apply_corridor_key(rgb_frame, alpha_2d)
                if profiler is not None:
                    profiler.sample("corridor", time.perf_counter() - corridor_start)
                    profiler.count("corridor")
            post_ms = (time.perf_counter() - post_start) * 1000.0

            compose_start = time.perf_counter()
            if not used_gpu_compose:
                output_final, self.pipeline_bg_cache, self.pipeline_last_bg_mode = self._compose_processed_frame(
                    compose_rgb_frame,
                    alpha_2d,
                    self.pipeline_bg_cache,
                    self.pipeline_last_bg_mode,
                )
            compose_done = time.perf_counter()
            if profiler is not None:
                profiler.sample("compose", compose_done - compose_start)
                profiler.count("compose")

            display_start = time.perf_counter()
            self._set_latest_display(rgb_frame, alpha_2d, output_final)
            if profiler is not None:
                profiler.sample("display", time.perf_counter() - display_start)
                profiler.count("display")

            self._queue_output_frame(frame_id, rgb_frame, alpha_2d, output_final)

            compose_ms = (compose_done - compose_start) * 1000.0
            done_time = time.perf_counter()
            total_ms = (done_time - frame_start) * 1000.0
            compute_ms = max(0.0, preprocess_ms + infer_ms + post_ms + compose_ms)
            if profiler is not None:
                profiler.sample("total", total_ms / 1000.0)
                profiler.sample("compute_total", compute_ms / 1000.0)
                profiler.count("frame_done")
                profiler.flush_if_due()
            self._update_perf_metrics(original_shape, total_ms, compute_ms, preprocess_ms, infer_ms, post_ms, compose_ms)

        self.pipeline_stop_event.set()
        with self.output_condition:
            self.output_condition.notify_all()

    def _take_latest_output_frame(self):
        with self.output_condition:
            while (
                self.is_running
                and not self.pipeline_stop_event.is_set()
                and (self.output_latest_payload is None or self.output_latest_frame_id == self.output_last_taken_frame_id)
            ):
                self.output_condition.wait(0.1)
            if self.output_latest_payload is None or self.output_latest_frame_id == self.output_last_taken_frame_id:
                return None
            self.output_last_taken_frame_id = self.output_latest_frame_id
            return self.output_latest_payload

    def video_output_loop(self):
        while self.is_running and not self.pipeline_stop_event.is_set():
            item = self._take_latest_output_frame()
            if item is None:
                continue
            frame_id, rgb_frame, alpha_2d, output_final = item
            profiler = self.pipeline_profiler
            output_start = time.perf_counter()
            wrote_output = self.write_live_output_frame(rgb_frame, alpha_2d, output_final)
            if profiler is not None:
                profiler.sample("decklink_output", time.perf_counter() - output_start)
                if wrote_output:
                    profiler.count("decklink_output")
