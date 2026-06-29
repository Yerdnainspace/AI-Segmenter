import cv2
import numpy as np


def ensure_odd_ksize(k: int, min_k: int = 1) -> int:
    k = int(k)
    if k < min_k:
        k = min_k
    if k % 2 == 0:
        k += 1
    return k


def keep_largest_component(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    mask_u8: 0/255 uint8
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8, copy=False)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_u8

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    if int(stats[best_idx, cv2.CC_STAT_AREA]) < int(min_area):
        return mask_u8

    out = np.zeros_like(mask_u8)
    out[labels == best_idx] = 255
    return out


def guided_refine_alpha(rgb_u8: np.ndarray, alpha_f32: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Edge-aware refine of alpha using guided filter (OpenCV ximgproc).
    """
    try:
        guide = rgb_u8.astype(np.float32) / 255.0
        src = alpha_f32.astype(np.float32, copy=False)
        refined = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=int(radius), eps=float(eps))
        return np.clip(refined, 0.0, 1.0).astype(np.float32, copy=False)
    except Exception:
        a = (alpha_f32 * 255.0).astype(np.uint8)
        a = cv2.bilateralFilter(a, d=5, sigmaColor=40, sigmaSpace=6)
        return (a.astype(np.float32) / 255.0).clip(0.0, 1.0)


def despill_green(rgb_u8: np.ndarray, alpha_f32: np.ndarray) -> np.ndarray:
    """
    Simple green-spill suppression near edges (for preview/greenscreen mode).
    """
    rgb = rgb_u8.astype(np.float32)
    a = alpha_f32.astype(np.float32)

    edge = (a > 0.15) & (a < 0.95)
    if not np.any(edge):
        return rgb_u8

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    rb_max = np.maximum(r, b)
    excess = g - rb_max
    excess = np.maximum(excess, 0.0)

    strength = 0.65
    g2 = g.copy()
    g2[edge] = g[edge] - excess[edge] * strength
    rgb[:, :, 1] = np.clip(g2, 0.0, 255.0)
    return rgb.astype(np.uint8)


def center_crop_resize_rgb(image_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize by cover-fit and center crop. This preserves aspect ratio instead of squeezing.
    """
    src_h, src_w = image_rgb.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, (resized_w - target_w) // 2)
    y0 = max(0, (resized_h - target_h) // 2)
    return resized[y0:y0 + target_h, x0:x0 + target_w].copy()


def generate_checker_background(width: int, height: int, tile: int = 40) -> np.ndarray:
    y_idx, x_idx = np.indices((height, width))
    pattern = ((x_idx // tile + y_idx // tile) % 2).astype(np.uint8)
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark = np.array([90, 90, 90], dtype=np.uint8)
    return np.where(pattern[..., None] == 0, light, dark)

