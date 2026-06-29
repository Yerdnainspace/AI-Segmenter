import threading

import numpy as np

from ai_segmenter.yolo_controls import YoloControlsMixin


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class YoloHarness(YoloControlsMixin):
    def __init__(self):
        self.yolo_enabled = Value(True)
        self.model_name = Value("MediaPipe Selfie")
        self.loaded_model_name = "MediaPipe Selfie"
        self.yolo_model_lock = threading.Lock()
        self.yolo_post_detector = object()
        self.yolo_detections = []
        self.yolo_selected_keys = set()


def test_yolo_detection_signature_uses_stable_key_and_class_id():
    harness = YoloHarness()
    detections = [
        {"key": "person 1", "class_id": 0, "conf": 0.9},
        {"key": "cup 1", "class_id": 41, "conf": 0.7},
    ]

    assert harness._yolo_detection_signature(detections) == (("person 1", 0), ("cup 1", 41))


def test_make_yolo_alpha_uses_selected_detection_mask():
    harness = YoloHarness()
    mask = np.zeros((3, 4), dtype=np.float32)
    mask[1, 2] = 1.0
    harness.yolo_detections = [
        {"key": "person 1", "box": (0, 0, 1, 1), "mask": mask},
        {"key": "cup 1", "box": (1, 1, 3, 3), "mask": np.ones((3, 4), dtype=np.float32)},
    ]
    harness.yolo_selected_keys = {"person 1"}

    alpha = harness._make_yolo_alpha((3, 4, 3))

    assert alpha.shape == (3, 4)
    assert alpha[1, 2] > 0.0
    assert alpha[1, 2] > alpha[0, 0]


def test_make_yolo_roi_returns_none_when_disabled():
    harness = YoloHarness()
    harness.yolo_enabled = Value(False)
    harness.yolo_detections = [{"key": "person 1", "box": (0, 0, 4, 4)}]
    harness.yolo_selected_keys = {"person 1"}

    assert harness._make_yolo_roi((8, 8, 3)) is None


def test_apply_yolo_roi_to_alpha_limits_unselected_regions():
    harness = YoloHarness()
    harness.yolo_detections = [
        {"key": "person 1", "box": (20, 20, 24, 24), "mask": None},
    ]
    harness.yolo_selected_keys = {"person 1"}
    alpha = np.ones((40, 40), dtype=np.float32)

    result = harness._apply_yolo_roi_to_alpha(alpha)

    assert result.shape == alpha.shape
    assert result[22, 22] > 0.0
    assert result[0, 0] == 0.0
