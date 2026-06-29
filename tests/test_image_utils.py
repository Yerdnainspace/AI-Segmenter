import numpy as np

from ai_segmenter.image_utils import (
    center_crop_resize_rgb,
    ensure_odd_ksize,
    generate_checker_background,
    keep_largest_component,
)


def test_ensure_odd_ksize_rounds_up_even_values():
    assert ensure_odd_ksize(4) == 5
    assert ensure_odd_ksize(5) == 5


def test_ensure_odd_ksize_respects_minimum():
    assert ensure_odd_ksize(0, min_k=3) == 3


def test_keep_largest_component_removes_smaller_components():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[0:1, 0:1] = 255
    mask[2:5, 3:7] = 255

    result = keep_largest_component(mask, min_area=1)

    assert result[0, 0] == 0
    assert result[3, 4] == 255
    assert int((result > 0).sum()) == 12


def test_center_crop_resize_rgb_returns_target_shape():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    result = center_crop_resize_rgb(image, 16, 16)
    assert result.shape == (16, 16, 3)


def test_generate_checker_background_uses_two_colors():
    checker = generate_checker_background(8, 8, tile=4)
    colors = {tuple(pixel) for row in checker for pixel in row}
    assert colors == {(220, 220, 220), (90, 90, 90)}

