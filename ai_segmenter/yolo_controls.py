import threading
import time

import cv2
import customtkinter as ctk
import numpy as np

from ai_segmenter.config import YOLO_POSTPROCESS_MODEL
from ai_segmenter.models import YoloObjectDetector


class YoloControlsMixin:
    def _resolve_yolo_force_device(self, for_postprocess=False):
        mode = self.yolo_device_mode.get()
        if mode == "CPU":
            return "cpu"
        if mode in ("Automatisch", "TensorRT", "CUDA", "CUDA Tensor"):
            return None
        return None

    def _yolo_prefer_tensorrt(self):
        return self.yolo_device_mode.get() != "CPU"

    def _yolo_device_mode_note(self, for_postprocess=False):
        mode = self.yolo_device_mode.get()
        if mode in ("Automatisch", "TensorRT", "CUDA", "CUDA Tensor"):
            return "TensorRT"
        return mode

    def _reset_yolo_runtime_state(self, clear_post_detector=False, clear_primary_detector=False):
        with self.yolo_async_lock:
            self.yolo_async_pending_frame = None
        with self.yolo_model_lock:
            if clear_post_detector:
                self.yolo_post_detector = None
            if clear_primary_detector:
                self.yolo_detector = None
        self.yolo_detections = []
        self.yolo_selected_keys.clear()
        self.yolo_seen_keys.clear()
        self.yolo_last_detection_signature = None
        self.yolo_last_ui_update_time = 0.0
        self.yolo_frame_counter = 0
        self.yolo_last_detect_time = -999.0
        self.yolo_last_analysis_ms = 0.0
        self.yolo_last_analysis_time = 0.0
        self._refresh_yolo_detection_list()
        self.refresh_display_view()

    def toggle_yolo(self):
        if not self.yolo_enabled.get():
            if self._is_yolo_primary_model():
                self.yolo_status.set("YOLO als Hauptmodell aktiv")
            else:
                self.yolo_status.set("YOLO-Nachbearbeitung aus")
                self._reset_yolo_runtime_state(clear_post_detector=True)
            return
        self.yolo_frame_counter = 0
        self.yolo_last_detect_time = -999.0
        self.yolo_status.set(f"YOLO-Nachbearbeitung wird geladen ({self._yolo_device_mode_note(True)}) ...")
        threading.Thread(target=self._load_yolo_post_worker, daemon=True).start()

    def change_yolo_model(self, choice):
        self._reset_yolo_runtime_state(clear_post_detector=self.yolo_enabled.get(), clear_primary_detector=True)
        if self._is_yolo_primary_model():
            self.yolo_status.set(f"YOLO wird geladen: {choice} ({self._yolo_device_mode_note(False)})")
            threading.Thread(target=self._load_yolo_worker, args=(choice,), daemon=True).start()
        elif self.yolo_enabled.get():
            self.yolo_status.set(f"YOLO-Nachbearbeitung nutzt {YOLO_POSTPROCESS_MODEL} ({self._yolo_device_mode_note(True)})")
            threading.Thread(target=self._load_yolo_post_worker, daemon=True).start()

    def change_yolo_device(self, choice):
        if choice in ("CUDA", "CUDA Tensor"):
            self.yolo_device_mode.set("TensorRT")
            choice = "TensorRT"
        self._reset_yolo_runtime_state(clear_post_detector=True, clear_primary_detector=True)
        if self._is_yolo_primary_model():
            model_name = self.yolo_model_name.get()
            self.yolo_status.set(f"YOLO wird auf {self._yolo_device_mode_note(False)} geladen: {model_name}")
            threading.Thread(target=self._load_yolo_worker, args=(model_name,), daemon=True).start()
        elif self.yolo_enabled.get():
            self.yolo_status.set(f"YOLO-Nachbearbeitung wird auf {self._yolo_device_mode_note(True)} geladen ...")
            threading.Thread(target=self._load_yolo_post_worker, daemon=True).start()
        else:
            self.yolo_status.set(f"YOLO aus / Hardware: {choice}")

    def _load_yolo_worker(self, model_name):
        try:
            use_yolo_trt = self.model_name.get() == "YOLO TensorRT" or self.loaded_model_name == "YOLO TensorRT"
            detector = YoloObjectDetector(
                model_name,
                force_device=self._resolve_yolo_force_device(False),
                prefer_tensorrt=use_yolo_trt,
                build_tensorrt_engine=use_yolo_trt,
            )
            with self.yolo_model_lock:
                self.yolo_detector = detector
            status = f"YOLO bereit: {model_name} ({detector.backend_label})"
            if detector.device_hint:
                status += "\n" + str(detector.device_hint)
            self.model_status = f"Modell bereit: YOLO ({detector.backend_label})"
            self.root.after(0, lambda: self.yolo_status.set(status))
            self.root.after(0, lambda: self.model_status_label.configure(text=self.model_status))
        except Exception as exc:
            with self.yolo_model_lock:
                self.yolo_detector = None
            self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Fehler: {e}"))

    def _load_yolo_post_worker(self):
        try:
            detector = YoloObjectDetector(
                YOLO_POSTPROCESS_MODEL,
                force_device=self._resolve_yolo_force_device(True),
                prefer_tensorrt=self._yolo_prefer_tensorrt(),
                build_tensorrt_engine=self._yolo_prefer_tensorrt(),
            )
            with self.yolo_model_lock:
                self.yolo_post_detector = detector
            status = f"YOLO-Nachbearbeitung bereit: {YOLO_POSTPROCESS_MODEL} ({detector.backend_label})"
            if detector.device_hint:
                status += "\n" + str(detector.device_hint)
            self.root.after(0, lambda: self.yolo_status.set(status))
            self.root.after(0, self._refresh_yolo_detection_list)
        except Exception as exc:
            with self.yolo_model_lock:
                self.yolo_post_detector = None
            self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Nachbearbeitung Fehler: {e}"))

    def _is_yolo_primary_model(self):
        return self.model_name.get() in ("YOLO", "YOLO TensorRT") or self.loaded_model_name in ("YOLO", "YOLO TensorRT")

    def _yolo_selection_active(self):
        return self.yolo_enabled.get() or self._is_yolo_primary_model()

    def _refresh_yolo_detection_list(self):
        if not hasattr(self, "yolo_objects_frame"):
            return
        for child in self.yolo_objects_frame.winfo_children():
            child.destroy()
        self.yolo_detection_vars = {}

        if not self._yolo_selection_active():
            self.yolo_objects_frame.pack_forget()
            return
        pack_options = {"pady": (0, 8), "padx": 10, "fill": "x"}
        if hasattr(self, "corridor_header"):
            pack_options["before"] = self.corridor_header
        elif hasattr(self, "metrics_header"):
            pack_options["before"] = self.metrics_header
        self.yolo_objects_frame.pack(**pack_options)
        if not self.yolo_detections:
            with self.yolo_model_lock:
                detector_ready = self.yolo_detector is not None if self._is_yolo_primary_model() else self.yolo_post_detector is not None
            text = "Keine Objekte erkannt" if detector_ready else "YOLO wird geladen ..."
            ctk.CTkLabel(
                self.yolo_objects_frame,
                text=text,
                justify="left",
                wraplength=280
            ).pack(anchor="w", fill="x")
            return

        current_keys = {det["key"] for det in self.yolo_detections}
        self.yolo_selected_keys.intersection_update(current_keys)
        if self.yolo_select_all.get():
            new_keys = current_keys - self.yolo_seen_keys
            self.yolo_selected_keys.update(new_keys)
        self.yolo_seen_keys = current_keys

        for det in self.yolo_detections:
            key = det["key"]
            var = ctk.BooleanVar(value=key in self.yolo_selected_keys)
            self.yolo_detection_vars[key] = var
            label = f"{key} ({det['conf']:.0%})"
            box = ctk.CTkCheckBox(
                self.yolo_objects_frame,
                text=label,
                variable=var,
                command=lambda k=key: self._toggle_yolo_detection(k)
            )
            box.pack(pady=2, anchor="w")

    def _toggle_yolo_detection(self, key):
        var = self.yolo_detection_vars.get(key)
        if var is not None and bool(var.get()):
            self.yolo_selected_keys.add(key)
        else:
            self.yolo_selected_keys.discard(key)

    def _yolo_detection_signature(self, detections):
        return tuple(
            (
                det.get("key"),
                int(det.get("class_id", -1)),
            )
            for det in detections
        )

    def _apply_yolo_detections(self, detections, elapsed_ms, now=None):
        now = time.perf_counter() if now is None else now
        self.yolo_detections = detections
        self.yolo_last_analysis_ms = float(elapsed_ms)
        self.yolo_last_analysis_time = now
        signature = self._yolo_detection_signature(detections)
        current_keys = {det["key"] for det in detections}
        self.yolo_selected_keys.intersection_update(current_keys)
        if self.yolo_select_all.get():
            new_keys = current_keys - self.yolo_seen_keys
            self.yolo_selected_keys.update(new_keys)
        self.yolo_seen_keys = current_keys
        selected_count = len([det for det in detections if det["key"] in self.yolo_selected_keys])
        ui_due = now - self.yolo_last_ui_update_time >= float(self.yolo_ui_update_interval)
        try:
            list_visible = bool(self.yolo_objects_frame.winfo_ismapped())
        except Exception:
            list_visible = True
        if signature != self.yolo_last_detection_signature or not list_visible:
            self.yolo_last_detection_signature = signature
            self.yolo_last_ui_update_time = now
            self.root.after(0, self._refresh_yolo_detection_list)
            ui_due = False
        if ui_due:
            self.yolo_last_ui_update_time = now
            self.root.after(
                0,
                lambda count=len(detections), ms=elapsed_ms, selected=selected_count: self.yolo_status.set(
                    f"YOLO aktiv: {count} Objekte, {selected} ausgewaehlt\nAnalyse: {ms:.1f} ms"
                )
            )
        return elapsed_ms

    def _run_yolo_primary(self, rgb_frame):
        with self.yolo_model_lock:
            detector = self.yolo_detector
        if detector is None:
            return 0.0
        self.yolo_frame_counter += 1
        now = time.perf_counter()
        self.yolo_last_detect_time = now
        start = time.perf_counter()
        detections = detector.detect(rgb_frame, conf=float(self.yolo_confidence.get()), max_det=8, imgsz=320)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._apply_yolo_detections(detections, elapsed_ms, now=now)

    def _schedule_yolo_postprocess(self, rgb_frame):
        if not self.yolo_enabled.get() or self._is_yolo_primary_model():
            return 0.0
        with self.yolo_model_lock:
            detector = self.yolo_post_detector
        if detector is None:
            return 0.0

        self.yolo_frame_counter += 1
        now = time.perf_counter()
        if self.yolo_frame_counter % int(self.yolo_detect_every) != 1:
            return 0.0
        if now - self.yolo_last_detect_time < float(self.yolo_min_detect_interval):
            return 0.0
        self.yolo_last_detect_time = now

        with self.yolo_async_lock:
            self.yolo_async_pending_frame = rgb_frame.copy()
            if self.yolo_async_running:
                return 0.0
            self.yolo_async_running = True

        self.yolo_async_thread = threading.Thread(target=self._yolo_postprocess_worker, daemon=True)
        self.yolo_async_thread.start()
        return 0.0

    def _run_yolo_postprocess_sync(self, rgb_frame):
        if not self.yolo_enabled.get() or self._is_yolo_primary_model():
            return 0.0
        with self.yolo_model_lock:
            detector = self.yolo_post_detector
        if detector is None:
            return 0.0
        self.yolo_frame_counter += 1
        now = time.perf_counter()
        self.yolo_last_detect_time = now
        start = time.perf_counter()
        try:
            detections = detector.detect(rgb_frame, conf=float(self.yolo_confidence.get()), max_det=8, imgsz=320)
        except Exception as exc:
            self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Fehler: {e}"))
            detections = []
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._apply_yolo_detections(detections, elapsed_ms, now=now)

    def _yolo_postprocess_worker(self):
        while True:
            with self.yolo_async_lock:
                frame = self.yolo_async_pending_frame
                self.yolo_async_pending_frame = None
            if frame is None:
                with self.yolo_async_lock:
                    self.yolo_async_running = False
                return
            if self.yolo_async_stop:
                with self.yolo_async_lock:
                    self.yolo_async_running = False
                return

            with self.yolo_model_lock:
                detector = self.yolo_post_detector
            if detector is None:
                with self.yolo_async_lock:
                    self.yolo_async_running = False
                return

            start = time.perf_counter()
            try:
                detections = detector.detect(frame, conf=float(self.yolo_confidence.get()), max_det=8, imgsz=320)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self.yolo_status.set(f"YOLO Fehler: {e}"))
                detections = []
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.root.after(0, lambda d=detections, ms=elapsed_ms: self._apply_yolo_detections(d, ms))

    def _make_yolo_roi(self, shape):
        h, w = shape[:2]
        if not self.yolo_enabled.get():
            return None
        with self.yolo_model_lock:
            detector_ready = self.yolo_post_detector is not None
        if not detector_ready:
            return None
        if not self.yolo_detections:
            return None
        mask = np.zeros((h, w), dtype=np.float32)
        selected = set(self.yolo_selected_keys)
        for det in self.yolo_detections:
            if det["key"] not in selected:
                continue
            x1, y1, x2, y2 = det["box"]
            pad_x = max(8, int((x2 - x1) * 0.06))
            pad_y = max(8, int((y2 - y1) * 0.06))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            mask[y1:y2, x1:x2] = 1.0
        if np.max(mask) <= 0:
            return mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return np.clip(mask, 0.0, 1.0)

    def _apply_yolo_roi_to_alpha(self, alpha_2d):
        roi = self._make_yolo_roi(alpha_2d.shape)
        if roi is None:
            return alpha_2d
        return np.clip(alpha_2d * roi, 0.0, 1.0).astype(np.float32, copy=False)

    def _make_yolo_alpha(self, shape):
        h, w = shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        selected = set(self.yolo_selected_keys)
        for det in self.yolo_detections:
            if det["key"] not in selected:
                continue
            det_mask = det.get("mask")
            if det_mask is not None:
                if det_mask.shape[:2] != (h, w):
                    det_mask = cv2.resize(det_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = np.maximum(alpha, det_mask.astype(np.float32, copy=False))
                continue

            x1, y1, x2, y2 = det["box"]
            pad_x = max(8, int((x2 - x1) * 0.04))
            pad_y = max(8, int((y2 - y1) * 0.04))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            alpha[y1:y2, x1:x2] = 1.0

        if np.max(alpha) > 0:
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)

    def _draw_yolo_overlay(self, frame_rgb):
        if not self._yolo_selection_active() or not self.yolo_detections:
            return frame_rgb
        out = frame_rgb.copy()
        selected = set(self.yolo_selected_keys)
        for det in self.yolo_detections:
            x1, y1, x2, y2 = det["box"]
            is_selected = det["key"] in selected
            color = (60, 220, 80) if is_selected else (235, 170, 60)
            thickness = 3 if is_selected else 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{det['key']} {det['conf']:.0%}"
            cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return out
