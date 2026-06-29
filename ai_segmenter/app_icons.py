import ctypes
import os
import sys

import customtkinter as ctk
from PIL import Image


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(APP_DIR, "assets")


def set_windows_app_id(app_id):
    if not sys.platform.startswith("win"):
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def _make_program_icon_image():
    customtkinter_icon = os.path.join(
        os.path.dirname(ctk.__file__),
        "assets",
        "icons",
        "CustomTkinter_icon_Windows.ico",
    )
    if os.path.exists(customtkinter_icon):
        return Image.open(customtkinter_icon).convert("RGBA").resize((256, 256), Image.LANCZOS)
    return Image.new("RGBA", (256, 256), (0, 120, 215, 255))


def apply_window_icon(window):
    try:
        from PIL import ImageTk

        os.makedirs(ASSET_DIR, exist_ok=True)
        icon_img = _make_program_icon_image()
        icon_path = os.path.join(ASSET_DIR, "ai_segmenter_program.ico")
        icon_img.save(
            icon_path,
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        window.iconbitmap(icon_path)
        icon_photo = ImageTk.PhotoImage(icon_img)
        window.iconphoto(True, icon_photo)
        window._app_icon_photo = icon_photo
    except Exception:
        pass

