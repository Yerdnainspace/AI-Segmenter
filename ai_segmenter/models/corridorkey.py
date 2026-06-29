import importlib.util
import os

import numpy as np

from ai_segmenter.app_icons import APP_DIR
from ai_segmenter.config import (
    CORRIDORKEY_CHECKPOINT_FILE,
    CORRIDORKEY_CHECKPOINT_REPO,
    CORRIDORKEY_IMG_SIZE,
)
from ai_segmenter.runtime import quiet_terminal_output


class CorridorKeyRefiner:
    def __init__(self, device_mode="Automatisch", img_size=CORRIDORKEY_IMG_SIZE):
        required_modules = ["torch", "timm", "safetensors", "huggingface_hub", "CorridorKeyModule"]
        missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if missing_modules:
            raise RuntimeError(
                "CorridorKey benoetigt zusaetzliche Python-Pakete/Module. "
                f"Fehlend: {', '.join(missing_modules)}. "
                "Bitte den Windows-Installer erneut ausfuehren."
            )

        import torch
        from huggingface_hub import hf_hub_download
        from CorridorKeyModule import CorridorKeyEngine
        from CorridorKeyModule.core import color_utils as corridor_color

        self.torch = torch
        self.corridor_color = corridor_color
        self.device_mode = device_mode
        self.device = self._resolve_device(torch, device_mode)
        self.device_label = "CUDA" if self.device == "cuda" else "CPU"
        self.img_size = int(img_size)

        checkpoint_dir = os.path.join(APP_DIR, "CorridorKeyModule", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, CORRIDORKEY_CHECKPOINT_FILE)
        if not os.path.exists(checkpoint_path):
            checkpoint_path = hf_hub_download(
                repo_id=CORRIDORKEY_CHECKPOINT_REPO,
                filename=CORRIDORKEY_CHECKPOINT_FILE,
                local_dir=checkpoint_dir,
            )

        os.environ.setdefault("CORRIDORKEY_SKIP_COMPILE", "1")
        with quiet_terminal_output():
            self.engine = CorridorKeyEngine(
                checkpoint_path=checkpoint_path,
                device=self.device,
                img_size=self.img_size,
                mixed_precision=self.device == "cuda",
            )

    @staticmethod
    def _resolve_device(torch, device_mode):
        if device_mode == "CPU":
            return "cpu"
        if device_mode == "CUDA":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA wurde fuer CorridorKey gewaehlt, ist in PyTorch aber nicht verfuegbar.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def refine(self, rgb_frame, alpha_2d, despill_strength=0.7, despeckle_size=400):
        mask = np.clip(alpha_2d.astype(np.float32, copy=False), 0.0, 1.0)
        despill_strength = float(np.clip(float(despill_strength), 0.0, 1.0))
        despeckle_size = max(0, int(despeckle_size))
        result = self.engine.process_frame(
            rgb_frame,
            mask,
            input_is_linear=False,
            despill_strength=despill_strength,
            auto_despeckle=True,
            despeckle_size=despeckle_size,
            generate_comp=False,
            post_process_on_gpu=self.device == "cuda",
            screen_channel=1,
        )
        processed_rgba = result.get("processed")
        refined_alpha = result.get("alpha")
        refined_fg = result.get("fg")
        if processed_rgba is not None:
            processed_rgba = np.nan_to_num(processed_rgba, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
            if processed_rgba.ndim == 3 and processed_rgba.shape[2] >= 4:
                processed_alpha = np.clip(processed_rgba[:, :, 3], 0.0, 1.0)
                premul_linear_rgb = np.clip(processed_rgba[:, :, :3], 0.0, None)
                straight_linear_rgb = premul_linear_rgb / np.maximum(processed_alpha[:, :, np.newaxis], 1e-4)
                refined_fg = self.corridor_color.linear_to_srgb(np.clip(straight_linear_rgb, 0.0, 1.0))
                refined_alpha = processed_alpha
        if refined_alpha is None:
            return rgb_frame, alpha_2d
        if refined_alpha.ndim == 3:
            refined_alpha = refined_alpha[:, :, 0]
        refined_alpha = np.nan_to_num(refined_alpha, nan=0.0, posinf=1.0, neginf=0.0)
        refined_alpha = np.clip(refined_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        if refined_fg is None:
            return rgb_frame, refined_alpha
        refined_fg = np.nan_to_num(refined_fg, nan=0.0, posinf=1.0, neginf=0.0)
        if refined_fg.dtype != np.uint8:
            refined_fg = np.clip(refined_fg * 255.0, 0, 255).astype(np.uint8)
        return refined_fg, refined_alpha

