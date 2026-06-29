import cv2
import numpy as np

from ai_segmenter.runtime import quiet_terminal_output, select_torch_device


class RVMByteDanceModel:
    def __init__(self, force_device=None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "RVM benoetigt PyTorch. Installiere z.B.: "
                "pip install torch torchvision"
            ) from exc

        self.torch = torch
        if force_device == "cpu":
            self.device = torch.device("cpu")
            self.device_label = "CPU"
            self.device_hint = "RVM wurde manuell auf CPU gesetzt. Das entlastet die GPU, ist aber langsamer."
        elif force_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer RVM gewaehlt, ist in PyTorch aber nicht verfuegbar.")
            self.device = torch.device("cuda")
            self.device_label = "CUDA"
            self.device_hint = None
        else:
            self.device, self.device_label, self.device_hint = select_torch_device(torch)
        self.rec = [None, None, None, None]
        with quiet_terminal_output():
            self.model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                "mobilenetv3",
                pretrained=True,
                trust_repo=True,
                verbose=False,
            )
        self.model.to(self.device)
        self.model.eval()

        self.use_autocast = self.device_label == "CUDA"
        if self.device_label in ("CUDA", "DirectML"):
            try:
                self.model = self.model.half()
            except Exception:
                pass

    def predict_mask(self, rgb_frame):
        torch = self.torch
        h, w = rgb_frame.shape[:2]
        scale = min(1.0, 512.0 / max(h, w))
        model_w = max(32, int(w * scale) // 32 * 32)
        model_h = max(32, int(h * scale) // 32 * 32)
        resized = cv2.resize(rgb_frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, pha, *self.rec = self.model(tensor, *self.rec, downsample_ratio=0.25)
            else:
                _, pha, *self.rec = self.model(tensor, *self.rec, downsample_ratio=0.25)

        mask = pha[0, 0].detach().float().cpu().numpy()
        return np.clip(mask * 255.0, 0, 255).astype(np.uint8)

