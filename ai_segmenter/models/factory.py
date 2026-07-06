from .birefnet import BiRefNetModel
from .mediapipe_selfie import MediaPipeSelfieModel
from .rvm import RVMByteDanceModel
from .vitmatte import ViTMatteModel


def create_segmentation_model(model_name, mediapipe_model_path, force_device=None):
    if model_name == "MediaPipe Selfie":
        return MediaPipeSelfieModel(mediapipe_model_path)
    if model_name == "BiRefNet":
        return BiRefNetModel(force_device=force_device)
    if model_name == "BiRefNet TensorRT":
        return BiRefNetModel(force_device=force_device, use_tensorrt=True)
    if model_name == "ViTMatte":
        return ViTMatteModel(force_device=force_device)
    if model_name == "ViTMatte TensorRT":
        return ViTMatteModel(force_device=force_device, use_tensorrt=True)
    if model_name == "RVM ByteDance":
        return RVMByteDanceModel(force_device=force_device)
    raise ValueError(f"Unbekanntes Modell: {model_name}")
