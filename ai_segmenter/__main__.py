import customtkinter as ctk

from .app import FoolproofSyncApp
from .app_icons import set_windows_app_id


def main():
    set_windows_app_id("AIObjectSeg.Segmenter")
    root = ctk.CTk()
    app = FoolproofSyncApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
