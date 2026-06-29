import cv2
import numpy as np
from PIL import Image

from ai_segmenter.image_utils import (
    center_crop_resize_rgb as _center_crop_resize_rgb,
    ensure_odd_ksize as _ensure_odd_ksize,
    generate_checker_background as _generate_checker_background,
)


class PreviewRendererMixin:
    def _postprocess_to_alpha(self, rgb_frame: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        Convert raw model mask to a stable, refined alpha matte (float32 [0,1]).
        Operates in UI resolution.
        """
        h, w = rgb_frame.shape[:2]
        if mask_u8.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha_raw = mask_u8.astype(np.float32) / 255.0
        if self.fast_live_alpha.get() and self.app_mode.get() == "Live":
            soft_size = _ensure_odd_ksize(int(self.edge_soft.get()), min_k=1)
            if soft_size > 1:
                alpha_raw = cv2.GaussianBlur(alpha_raw, (soft_size, soft_size), 0)
            erode_size = int(self.edge_erode.get())
            if erode_size > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
                gate = cv2.erode((mask_u8 >= 128).astype(np.uint8), k, iterations=1).astype(np.float32)
                alpha_raw *= gate
            return np.clip(alpha_raw, 0.0, 1.0)

        bin_thresh = 128
        bin_mask = ((mask_u8 >= bin_thresh).astype(np.uint8) * 255)

        k_close = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        erode_size = int(self.edge_erode.get())
        if erode_size > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            bin_mask = cv2.erode(bin_mask, k, iterations=1)

        soft_size = _ensure_odd_ksize(int(self.edge_soft.get()), min_k=1)
        alpha_core = cv2.GaussianBlur(bin_mask, (soft_size, soft_size), 0).astype(np.float32) / 255.0

        core_gate = (alpha_core > 0.03).astype(np.float32)
        alpha = np.clip(alpha_raw * core_gate, 0.0, 1.0)
        alpha = np.maximum(alpha, alpha_core * 0.85)

        return alpha

    def update_bg_button_state(self, *args):
        if self.bg_mode.get() == "CustomImage":
            self.btn_load_bg.configure(state="normal")
        else:
            self.btn_load_bg.configure(state="disabled")

    def _get_checker_background(self, width: int, height: int) -> np.ndarray:
        if self.checker_background_source is None:
            self.checker_background_source = _generate_checker_background(1920, 1080)
        return _center_crop_resize_rgb(self.checker_background_source, width, height)

    def _get_custom_background(self, width: int, height: int) -> np.ndarray:
        if self.custom_background_source is None:
            return np.zeros((height, width, 3), dtype=np.uint8) + 30
        return _center_crop_resize_rgb(self.custom_background_source, width, height)

    def _alpha_to_u8(self, alpha_2d: np.ndarray) -> np.ndarray:
        if isinstance(alpha_2d, np.ndarray) and alpha_2d.dtype == np.uint8:
            return alpha_2d
        return np.clip(alpha_2d * 255.0, 0, 255).astype(np.uint8)

    def _make_alpha_preview(self, alpha_2d: np.ndarray) -> Image.Image:
        alpha_u8 = self._alpha_to_u8(alpha_2d)
        return Image.fromarray(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB))

    def _make_display_image(self, rgb_frame: np.ndarray, alpha_2d: np.ndarray, processed_frame: np.ndarray) -> Image.Image:
        view_mode = self.view_mode.get()
        if view_mode == "Input":
            return Image.fromarray(self._draw_yolo_overlay(rgb_frame))
        if view_mode == "Alpha Matte":
            return self._make_alpha_preview(alpha_2d)
        if processed_frame.shape[2] == 4:
            return Image.fromarray(processed_frame)
        return Image.fromarray(processed_frame)

    def _make_view_frame(self, rgb_frame: np.ndarray, alpha_2d: np.ndarray, processed_frame: np.ndarray) -> np.ndarray:
        view_mode = self.view_mode.get()
        if view_mode == "Input":
            return rgb_frame
        if view_mode == "Alpha Matte":
            alpha_u8 = self._alpha_to_u8(alpha_2d)
            return cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB)
        return processed_frame

    def _set_latest_display(self, rgb_frame: np.ndarray, alpha_2d: np.ndarray, processed_frame: np.ndarray):
        self.latest_display_payload = (rgb_frame, alpha_2d, processed_frame)
        self.latest_display_payload_id += 1

    def refresh_display_view(self, *args):
        payload = self.latest_display_payload
        if payload is None:
            return
        rgb_frame, alpha_2d, processed_frame = payload
        self.latest_pil_image = self._make_display_image(rgb_frame, alpha_2d, processed_frame)
        self.displayed_payload_id = self.latest_display_payload_id
