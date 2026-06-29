import os

from .app_icons import ASSET_DIR


MODEL_OPTIONS = ["MediaPipe Selfie", "BiRefNet", "BiRefNet TensorRT", "RVM ByteDance", "YOLO", "YOLO TensorRT"]
MAIN_AI_DEVICE_OPTIONS = ["Automatisch", "CUDA", "CPU"]
YOLO_MODEL_OPTIONS = ["yolo11n-seg.pt", "yolo11s-seg.pt"]
YOLO_DEVICE_OPTIONS = ["Automatisch", "TensorRT", "CPU"]
YOLO_POSTPROCESS_MODEL = "yolo11n.pt"
YOLO_TENSORRT_IMGSZ = 320
YOLO_TENSORRT_DIR = os.path.join(ASSET_DIR, "yolo_tensorrt")
BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet"
CORRIDORKEY_DEVICE_OPTIONS = ["Automatisch", "CUDA", "CPU"]
CORRIDORKEY_CHECKPOINT_REPO = "nikopueringer/CorridorKey_v1.0"
CORRIDORKEY_CHECKPOINT_FILE = "CorridorKey_v1.0.safetensors"
CORRIDORKEY_IMG_SIZE = 2048

DECKLINK_SDK_DLL_PATHS = [
    r"C:\Program Files (x86)\Blackmagic Design\Blackmagic Desktop Video\DeckLinkAPI.dll",
    r"C:\Program Files\Blackmagic Design\Blackmagic Desktop Video\DeckLinkAPI.dll",
    r"C:\Windows\System32\DeckLinkAPI.dll",
]

DECKLINK_OUTPUT_MODES = {
    "1080p25 - 1920x1080": ("bmdModeHD1080p25", 1920, 1080, 25.0),
    "1080p29.97 - 1920x1080": ("bmdModeHD1080p2997", 1920, 1080, 30000.0 / 1001.0),
    "1080p30 - 1920x1080": ("bmdModeHD1080p30", 1920, 1080, 30.0),
    "1080p50 - 1920x1080": ("bmdModeHD1080p50", 1920, 1080, 50.0),
    "1080p59.94 - 1920x1080": ("bmdModeHD1080p5994", 1920, 1080, 60000.0 / 1001.0),
    "1080p60 - 1920x1080": ("bmdModeHD1080p6000", 1920, 1080, 60.0),
    "720p50 - 1280x720": ("bmdModeHD720p50", 1280, 720, 50.0),
    "720p59.94 - 1280x720": ("bmdModeHD720p5994", 1280, 720, 60000.0 / 1001.0),
    "720p60 - 1280x720": ("bmdModeHD720p60", 1280, 720, 60.0),
}

