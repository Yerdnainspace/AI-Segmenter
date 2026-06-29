import importlib.util
import sys

import cv2
import numpy as np

from ai_segmenter.config import BIREFNET_REPO_ID
from ai_segmenter.runtime import prepare_tensorrt_import, quiet_terminal_output, select_torch_device


class BiRefNetModel:
    def __init__(self, force_device=None, use_tensorrt=False):
        required_modules = ["torch", "torchvision", "transformers", "timm", "safetensors"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "BiRefNet benoetigt zusaetzliche Python-Pakete. "
                f"Fehlend: {', '.join(missing_modules)}. "
                f"Installiere sie im aktiven Python mit: \"{sys.executable}\" -m pip install "
                "torch torchvision transformers timm safetensors"
            )

        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "BiRefNet benoetigt PyTorch und transformers. Installiere z.B.: "
                "pip install torch torchvision transformers timm safetensors"
            ) from exc

        self.torch = torch
        self.use_tensorrt = bool(use_tensorrt)
        self.tensorrt_enabled = False
        self.tensorrt_status = "TensorRT aus"
        self.input_size = 512
        if force_device == "cpu":
            if self.use_tensorrt:
                raise RuntimeError("BiRefNet TensorRT benoetigt CUDA. CPU wurde gewaehlt.")
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "BiRefNet wurde manuell auf CPU gesetzt. Das entlastet die GPU, ist aber langsamer."
        elif force_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer BiRefNet gewaehlt, ist in PyTorch aber nicht verfuegbar.")
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
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            from transformers import logging as transformers_logging
            transformers_logging.set_verbosity_error()
        except Exception:
            pass
        try:
            with quiet_terminal_output():
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    BIREFNET_REPO_ID,
                    trust_remote_code=True,
                )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                "BiRefNet konnte nicht geladen werden. Bitte den Installer erneut starten, "
                "damit die BiRefNet-Pakete und der HuggingFace-Cache repariert werden. "
                f"Originalfehler: {exc}"
            ) from exc

        if self.device_label != "CUDA":
            try:
                self.model = self.model.float()
            except Exception:
                pass

        self.use_autocast = False
        self.autocast_dtype = None
        if self.device_label == "CUDA":
            try:
                self.model = self.model.half()
                self.use_autocast = True
                self.autocast_dtype = torch.float16
            except Exception:
                pass

        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32

        if self.use_tensorrt:
            self._compile_tensorrt()

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
            dtype=self.model_dtype,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
            dtype=self.model_dtype,
        ).view(1, 3, 1, 1)

        if self.device_label != "CPU":
            try:
                dummy = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device, dtype=self.model_dtype)
                with torch.inference_mode():
                    _ = self.model(dummy)
            except Exception as exc:
                if self.device_label == "CUDA" and self.model_dtype == torch.float16:
                    try:
                        self.model = self.model.float()
                        self.model_dtype = next(self.model.parameters()).dtype
                        self.mean = self.mean.float()
                        self.std = self.std.float()
                        self.use_autocast = False
                        dummy = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device, dtype=self.model_dtype)
                        with torch.inference_mode():
                            _ = self.model(dummy)
                    except Exception as retry_exc:
                        raise RuntimeError(
                            "BiRefNet ist geladen, aber der CUDA-Warmup ist fehlgeschlagen. "
                            "Starte den Installer erneut, um PyTorch/torchvision passend zur GPU zu reparieren. "
                            f"Originalfehler: {retry_exc}"
                        ) from retry_exc
                else:
                    raise RuntimeError(
                        "BiRefNet ist geladen, aber der Warmup ist fehlgeschlagen. "
                        "Starte den Installer erneut, um die Installation zu reparieren. "
                        f"Originalfehler: {exc}"
                    ) from exc

    def _compile_tensorrt(self):
        if self.device_label != "CUDA":
            raise RuntimeError("BiRefNet TensorRT benoetigt CUDA.")
        try:
            prepare_tensorrt_import()
            import torch_tensorrt
        except ImportError as exc:
            raise RuntimeError(
                "BiRefNet TensorRT benoetigt torch-tensorrt und tensorrt. "
                "Bitte den Installer erneut ausfuehren."
            ) from exc

        torch = self.torch
        dummy = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device, dtype=self.model_dtype)
        try:
            with torch.inference_mode():
                self.model = torch_tensorrt.compile(
                    self.model,
                    ir="dynamo",
                    inputs=[torch_tensorrt.Input(dummy.shape, dtype=self.model_dtype)],
                    truncate_double=True,
                    require_full_compilation=False,
                    min_block_size=3,
                )
                _ = self.model(dummy)
            self.tensorrt_enabled = True
            self.tensorrt_status = "TensorRT aktiv"
            self.device_hint = "BiRefNet laeuft ueber Torch-TensorRT. Der erste Start kann wegen Engine-Build lange dauern."
        except Exception as exc:
            raise RuntimeError(
                "Torch-TensorRT konnte BiRefNet nicht kompilieren. "
                "Das Modell bleibt nicht automatisch auf PyTorch zurueck, damit der Fehler sichtbar bleibt. "
                f"Originalfehler: {exc}"
            ) from exc

    def _extract_prediction(self, output):
        torch = self.torch
        if isinstance(output, dict):
            for key in ("logits", "preds", "prediction", "out"):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if not torch.is_tensor(output):
            raise RuntimeError("BiRefNet hat kein Tensor-Ergebnis geliefert.")
        return output

    def predict_alpha_tensor(self, rgb_frame):
        torch = self.torch
        out_h, out_w = rgb_frame.shape[:2]
        resized = cv2.resize(rgb_frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.model_dtype) / 255.0
        tensor = (tensor - self.mean) / self.std

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    output = self.model(tensor)
            else:
                output = self.model(tensor)
            pred = self._extract_prediction(output).sigmoid()
            if pred.shape[-2:] != (out_h, out_w):
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                )

        return pred[0].detach().squeeze().clamp(0.0, 1.0)

    def predict_mask(self, rgb_frame):
        pred = self.predict_alpha_tensor(rgb_frame)
        mask = pred.detach().float().cpu().squeeze().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        return np.clip(mask * 255.0, 0, 255).astype(np.uint8)

