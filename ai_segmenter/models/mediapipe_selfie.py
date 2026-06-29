import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ai_segmenter.runtime import quiet_terminal_output


class MediaPipeSelfieModel:
    def __init__(self, model_path):
        self.device_label = "CPU"
        self.device_hint = "MediaPipe Selfie laeuft in diesem Programm CPU-basiert."
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        with quiet_terminal_output():
            self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def predict_mask(self, rgb_frame):
        ai_frame = cv2.resize(rgb_frame, (512, 288))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ai_frame)
        result = self.segmenter.segment(mp_image)
        mask_raw = result.category_mask.numpy_view()
        return (mask_raw > 0).astype(np.uint8) * 255

