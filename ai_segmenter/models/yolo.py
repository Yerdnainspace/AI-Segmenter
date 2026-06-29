import importlib.util
import os
import shutil
import sys

import cv2
import numpy as np

from ai_segmenter.config import YOLO_TENSORRT_DIR, YOLO_TENSORRT_IMGSZ
from ai_segmenter.runtime import prepare_tensorrt_import, quiet_terminal_output, select_torch_device


class YoloObjectDetector:
    def __init__(
        self,
        model_name="yolo11n-seg.pt",
        force_device=None,
        prefer_tensorrt=True,
        engine_imgsz=YOLO_TENSORRT_IMGSZ,
        build_tensorrt_engine=False,
    ):
        required_modules = ["ultralytics", "torch"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "YOLO benoetigt zusaetzliche Python-Pakete. "
                f"Fehlend: {', '.join(missing_modules)}. "
                f"Installiere sie mit: \"{sys.executable}\" -m pip install ultralytics"
            )

        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.yolo_cls = YOLO
        self.model_name = model_name
        self.force_device = force_device
        self.prefer_tensorrt = bool(prefer_tensorrt)
        self.engine_imgsz = int(engine_imgsz)
        self.build_tensorrt_engine = bool(build_tensorrt_engine)
        self.using_tensorrt = False
        if force_device == "cpu":
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "YOLO laeuft auf CPU. Das entlastet die GPU, ist aber langsamer als CUDA."
        else:
            self.device, self.device_label, self.device_hint = select_torch_device(torch)

        load_name = model_name
        self.backend_label = self.device_label
        if self.device_label == "CUDA" and self.prefer_tensorrt:
            try:
                load_name = self._ensure_tensorrt_engine(YOLO, model_name)
                self.using_tensorrt = True
                self.backend_label = "TensorRT"
                self.device_hint = "YOLO laeuft ueber TensorRT."
            except Exception as exc:
                self.device_hint = (
                    "YOLO TensorRT konnte nicht vorbereitet werden; fallback auf CUDA-PyTorch. "
                    f"Originalfehler: {exc}"
                )

        try:
            with quiet_terminal_output():
                self.model = YOLO(load_name)
                self.names = getattr(self.model, "names", {}) or {}
        except Exception as exc:
            if not self.using_tensorrt:
                raise
            self._quarantine_bad_engine(load_name)
            self.using_tensorrt = False
            self.backend_label = self.device_label
            self.device_hint = (
                "YOLO TensorRT-Engine konnte nicht geladen werden; fallback auf CUDA-PyTorch. "
                f"Originalfehler: {exc}"
            )
            with quiet_terminal_output():
                self.model = YOLO(model_name)
                self.names = getattr(self.model, "names", {}) or {}
        self.predict_device = "cpu" if force_device == "cpu" else (0 if self.device_label == "CUDA" else "cpu")

    def _engine_cache_path(self, model_name):
        stem = os.path.splitext(os.path.basename(str(model_name)))[0]
        suffix = f"imgsz{self.engine_imgsz}_fp16"
        return os.path.join(YOLO_TENSORRT_DIR, f"{stem}_{suffix}.engine")

    def _ensure_tensorrt_engine(self, yolo_cls, model_name):
        if not str(model_name).lower().endswith(".pt"):
            return model_name
        prepare_tensorrt_import()
        os.makedirs(YOLO_TENSORRT_DIR, exist_ok=True)
        engine_path = self._engine_cache_path(model_name)
        if os.path.exists(engine_path) and os.path.getsize(engine_path) > 1_000_000:
            return engine_path
        if not self.build_tensorrt_engine:
            raise RuntimeError(
                "keine vorbereitete YOLO TensorRT-Engine gefunden. "
                "Bitte Installer erneut ausfuehren."
            )
        with quiet_terminal_output():
            source_model = yolo_cls(model_name)
            exported_path = source_model.export(
                format="engine",
                imgsz=self.engine_imgsz,
                half=True,
                device=0,
                dynamic=False,
                verbose=False,
            )
        if exported_path and os.path.exists(exported_path):
            if os.path.abspath(exported_path) != os.path.abspath(engine_path):
                shutil.copy2(exported_path, engine_path)
            return engine_path
        default_engine = os.path.splitext(str(model_name))[0] + ".engine"
        if os.path.exists(default_engine):
            shutil.copy2(default_engine, engine_path)
            return engine_path
        raise RuntimeError("Ultralytics hat keine TensorRT-Engine-Datei erzeugt.")

    def _quarantine_bad_engine(self, engine_path):
        if not engine_path or not str(engine_path).lower().endswith(".engine"):
            return
        if not os.path.exists(engine_path):
            return
        bad_path = str(engine_path) + ".bad"
        try:
            if os.path.exists(bad_path):
                os.remove(bad_path)
            os.replace(engine_path, bad_path)
        except OSError:
            pass

    def detect(self, rgb_frame, conf=0.35, max_det=12, imgsz=416):
        try:
            results = self._predict(rgb_frame, conf, max_det, imgsz)
        except Exception as exc:
            if not self.using_tensorrt:
                raise
            self._fallback_from_tensorrt(exc)
            results = self._predict(rgb_frame, conf, max_det, imgsz)
        return self._parse_results(results, rgb_frame)

    def _predict(self, rgb_frame, conf, max_det, imgsz):
        predict_kwargs = {
            "source": rgb_frame,
            "conf": float(conf),
            "max_det": int(max_det),
            "imgsz": int(imgsz),
            "verbose": False,
        }
        if not self.using_tensorrt:
            predict_kwargs["device"] = self.predict_device
        with quiet_terminal_output():
            return self.model.predict(**predict_kwargs)

    def _fallback_from_tensorrt(self, exc):
        engine_path = self._engine_cache_path(self.model_name)
        self._quarantine_bad_engine(engine_path)
        self.using_tensorrt = False
        self.backend_label = self.device_label
        self.device_hint = (
            "YOLO TensorRT ist waehrend der Inferenz fehlgeschlagen; "
            f"fallback auf {self.device_label}-PyTorch. Originalfehler: {exc}"
        )
        with quiet_terminal_output():
            self.model = self.yolo_cls(self.model_name)
            self.names = getattr(self.model, "names", {}) or {}

    def _parse_results(self, results, rgb_frame):
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        masks_obj = getattr(result, "masks", None)
        masks = None
        if masks_obj is not None and getattr(masks_obj, "data", None) is not None:
            try:
                masks = masks_obj.data.detach().float().cpu().numpy()
            except Exception:
                masks = None
        detections = []
        h, w = rgb_frame.shape[:2]
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(float)
            x1 = int(max(0, min(w - 1, round(x1))))
            y1 = int(max(0, min(h - 1, round(y1))))
            x2 = int(max(0, min(w, round(x2))))
            y2 = int(max(0, min(h, round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(cls[i])
            name = str(self.names.get(class_id, class_id))
            mask = None
            if masks is not None and i < len(masks):
                mask = masks[i]
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
            detections.append(
                {
                    "name": name,
                    "class_id": class_id,
                    "conf": float(confs[i]),
                    "box": (x1, y1, x2, y2),
                    "mask": mask,
                }
            )

        detections.sort(key=lambda item: (item["name"], item["box"][0], item["box"][1]))
        counts = {}
        for det in detections:
            counts[det["name"]] = counts.get(det["name"], 0) + 1
            det["key"] = f"{det['name']} {counts[det['name']]}"
        return detections

