import numpy as np

from ai_segmenter.preview_renderer import PreviewRendererMixin


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class PreviewHarness(PreviewRendererMixin):
    def __init__(self):
        self.fast_live_alpha = Value(True)
        self.app_mode = Value("Live")
        self.edge_soft = Value(1)
        self.edge_erode = Value(0)
        self.view_mode = Value("Processed")
        self.checker_background_source = None
        self.custom_background_source = None
        self.latest_display_payload = None
        self.latest_display_payload_id = 0
        self.displayed_payload_id = -1
        self.latest_pil_image = None

    def _draw_yolo_overlay(self, frame_rgb):
        return frame_rgb


def test_alpha_to_u8_clips_float_alpha():
    harness = PreviewHarness()
    alpha = np.array([[-0.5, 0.5, 2.0]], dtype=np.float32)

    result = harness._alpha_to_u8(alpha)

    assert result.dtype == np.uint8
    assert result.tolist() == [[0, 127, 255]]


def test_postprocess_to_alpha_fast_live_returns_float_mask_shape():
    harness = PreviewHarness()
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[:, 1:] = 255

    alpha = harness._postprocess_to_alpha(rgb, mask)

    assert alpha.shape == (4, 5)
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0


def test_custom_background_falls_back_to_neutral_rgb_image():
    harness = PreviewHarness()

    result = harness._get_custom_background(3, 2)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.uint8
    assert np.all(result == 30)


def test_refresh_display_view_renders_latest_payload():
    harness = PreviewHarness()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    alpha = np.ones((2, 2), dtype=np.float32)
    processed = np.full((2, 2, 3), 200, dtype=np.uint8)
    harness._set_latest_display(rgb, alpha, processed)

    harness.refresh_display_view()

    assert harness.displayed_payload_id == 1
    assert harness.latest_pil_image.size == (2, 2)
