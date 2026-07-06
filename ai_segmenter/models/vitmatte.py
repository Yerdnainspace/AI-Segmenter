import contextlib
import importlib.util
import sys

import cv2
import numpy as np

from ai_segmenter.runtime import (
    TENSORRT_RUNTIME_LOCK,
    prepare_tensorrt_import,
    quiet_terminal_output,
    select_torch_device,
)


class ViTMatteModel:
    def __init__(self, force_device=None, use_tensorrt=False):
        required_modules = ["torch", "torchvision", "transformers", "timm", "safetensors", "einops", "kornia"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "ViTMatte benoetigt zusaetzliche Python-Pakete. "
                f"Fehlend: {', '.join(missing_modules)}. "
                f"Installiere sie im aktiven Python mit: \"{sys.executable}\" -m pip install "
                "torch torchvision transformers timm safetensors einops kornia"
            )

        try:
            import torch
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "ViTMatte benoetigt PyTorch und transformers. Installiere z.B.: "
                "pip install torch torchvision transformers timm safetensors einops kornia"
            ) from exc

        self.torch = torch
        self.processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-base-composition-1k")
        self.use_tensorrt = bool(use_tensorrt)
        self.tensorrt_enabled = False
        self.tensorrt_status = "TensorRT aus"
        self._pytorch_model = None
        self.input_size = 512

        if force_device == "cpu":
            if self.use_tensorrt:
                raise RuntimeError("ViTMatte TensorRT benoetigt CUDA. CPU wurde gewaehlt.")
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "ViTMatte wurde manuell auf CPU gesetzt. Das entlastet die GPU, ist aber langsamer."
        elif force_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer ViTMatte gewaehlt, ist in PyTorch aber nicht verfuegbar.")
            self.device = torch.device("cuda")
            self.device_label = "CUDA"
            self.device_hint = None
        else:
            self.device, self.device_label, self.device_hint = select_torch_device(torch)

        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        try:
            with quiet_terminal_output():
                self.model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-base-composition-1k")
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"ViTMatte konnte nicht geladen werden. Originalfehler: {exc}") from exc

        self.use_autocast = self.device_label == "CUDA"
        self.autocast_dtype = torch.float16 if self.use_autocast else None
        if self.device_label == "CUDA":
            try:
                self.model = self.model.half()
            except Exception:
                pass

        if self.use_tensorrt:
            self._compile_tensorrt()

    def _heuristic_trimap(self, rgb_frame):
        h, w = rgb_frame.shape[:2]
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
        _, bg = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
        fg = cv2.erode(fg, np.ones((7, 7), np.uint8), iterations=1)
        bg = cv2.dilate(bg, np.ones((11, 11), np.uint8), iterations=1)
        trimap = np.full((h, w), 128, dtype=np.uint8)
        trimap[bg > 0] = 0
        trimap[fg > 0] = 255
        return trimap

    def _prepare_inputs(self, rgb_frame):
        trimap = self._heuristic_trimap(rgb_frame)
        image = cv2.resize(rgb_frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        trimap = cv2.resize(trimap, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        return image, trimap

    def _forward(self, image_tensor, trimap_tensor):
        inputs = self.processor(images=image_tensor, trimaps=trimap_tensor, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            runtime_context = TENSORRT_RUNTIME_LOCK if self.tensorrt_enabled else contextlib.nullcontext()
            with runtime_context:
                if self.use_autocast:
                    with self.torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                        output = self.model(**inputs)
                else:
                    output = self.model(**inputs)
        alpha = getattr(output, "alphas", None)
        if alpha is None:
            alpha = getattr(output, "alpha", None)
        if alpha is None and isinstance(output, (list, tuple)):
            alpha = output[0]
        if alpha is None:
            raise RuntimeError("ViTMatte hat kein Alpha-Ergebnis geliefert.")
        return alpha

    def _compile_tensorrt(self):
        if self.device_label != "CUDA":
            raise RuntimeError("ViTMatte TensorRT benoetigt CUDA.")
        try:
            prepare_tensorrt_import()
            import torch_tensorrt
        except ImportError as exc:
            raise RuntimeError("ViTMatte TensorRT benoetigt torch-tensorrt und tensorrt.") from exc

        torch = self.torch
        dummy_img = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)
        dummy_trimap = torch.zeros((1, 1, self.input_size, self.input_size), device=self.device)
        pytorch_model = self.model
        try:
            with torch.inference_mode():
                with TENSORRT_RUNTIME_LOCK:
                    self.model = torch_tensorrt.compile(
                        self.model,
                        ir="dynamo",
                        inputs=[
                            torch_tensorrt.Input(dummy_img.shape, dtype=dummy_img.dtype),
                            torch_tensorrt.Input(dummy_trimap.shape, dtype=dummy_trimap.dtype),
                        ],
                        truncate_double=True,
                        require_full_compilation=False,
                        min_block_size=3,
                        use_python_runtime=True,
                    )
                    _ = self.model(dummy_img, dummy_trimap)
            self._pytorch_model = pytorch_model
            self.tensorrt_enabled = True
            self.tensorrt_status = "TensorRT aktiv"
            self.device_hint = "ViTMatte laeuft ueber Torch-TensorRT. Der erste Start kann laenger dauern."
        except Exception as exc:
            self.model = pytorch_model
            self._pytorch_model = None
            self.tensorrt_enabled = False
            self.tensorrt_status = "TensorRT Fehler"
            self.device_hint = (
                "Torch-TensorRT konnte ViTMatte nicht stabil vorbereiten; "
                f"fallback auf CUDA-PyTorch. Originalfehler: {exc}"
            )

    def predict_mask(self, rgb_frame):
        image, trimap = self._prepare_inputs(rgb_frame)
        torch = self.torch
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
        alpha = self._forward(image_tensor, trimap_tensor)
        if alpha.shape[-2:] != rgb_frame.shape[:2]:
            alpha = torch.nn.functional.interpolate(
                alpha,
                size=rgb_frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        alpha = alpha[0].detach().squeeze().clamp(0.0, 1.0)
        return np.clip(alpha.float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
