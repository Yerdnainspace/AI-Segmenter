import customtkinter as ctk

from ai_segmenter.camera import get_available_cameras
from ai_segmenter.config import (
    CORRIDORKEY_DEVICE_OPTIONS,
    DECKLINK_OUTPUT_MODES,
    MAIN_AI_DEVICE_OPTIONS,
    MODEL_OPTIONS,
    YOLO_DEVICE_OPTIONS,
    YOLO_MODEL_OPTIONS,
)
from ai_segmenter.decklink import get_decklink_output_devices, get_live_input_sources
from ai_segmenter.utils import run_with_timeout


class AppLayoutMixin:
    def setup_gui_legacy(self):
        self.video_label = ctk.CTkLabel(self.root, text="Kamera gestoppt", width=self.ui_w, height=self.ui_h,
                                        fg_color="#2b2b2b")
        self.video_label.pack(side="left", padx=20, pady=20, expand=True, fill="both")

        control_panel = ctk.CTkFrame(self.root, width=320)
        control_panel.pack(side="right", fill="y", padx=20, pady=20)

        title = ctk.CTkLabel(control_panel, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=15)

        ctk.CTkLabel(control_panel, text="AI-Modell", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 5))
        self.model_select = ctk.CTkOptionMenu(control_panel, values=MODEL_OPTIONS,
                                              variable=self.model_name,
                                              command=self.change_model)
        self.model_select.pack(pady=5, padx=10, fill="x")

        self.model_status_label = ctk.CTkLabel(control_panel, text=self.model_status, wraplength=280,
                                               justify="left")
        self.model_status_label.pack(pady=(0, 10), padx=10, fill="x")

        self.camera_select = ctk.CTkOptionMenu(control_panel, values=run_with_timeout(get_available_cameras, ["Keine Kamera gefunden"], timeout=6.0),
                                               command=self.change_camera)
        self.camera_select.pack(pady=10, padx=10, fill="x")

        self.btn_refresh_cameras = ctk.CTkButton(control_panel, text="Kameras neu suchen",
                                                command=self.refresh_cameras)
        self.btn_refresh_cameras.pack(pady=5, padx=10, fill="x")

        self.btn_toggle = ctk.CTkButton(control_panel, text="Kamera Starten", command=self.toggle_camera)
        self.btn_toggle.pack(pady=10, padx=10, fill="x")

        metrics_header = ctk.CTkFrame(control_panel, fg_color="transparent")
        metrics_header.pack(pady=(10, 5), padx=10, fill="x")
        ctk.CTkLabel(metrics_header, text="Live-Metriken", font=ctk.CTkFont(size=14, weight="bold")).pack(
            side="left")
        self.btn_metrics_info = ctk.CTkButton(
            metrics_header,
            text="i",
            width=26,
            height=24,
            command=self.show_metrics_info
        )
        self.btn_metrics_info.pack(side="right")
        self.metrics_label = ctk.CTkLabel(
            control_panel,
            text=self.metrics_text,
            font=ctk.CTkFont(family="Consolas", size=12),
            justify="left",
            anchor="w",
            wraplength=280
        )
        self.metrics_label.pack(pady=(0, 8), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Hintergrund-Effekt", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(20, 5))

        rb_checker = ctk.CTkRadioButton(control_panel, text="Karo-Muster (Transparenz)", variable=self.bg_mode,
                                        value="Checker")
        rb_checker.pack(pady=5, anchor="w", padx=20)

        rb_transparent = ctk.CTkRadioButton(control_panel, text="Echt Transparent", variable=self.bg_mode,
                                            value="Transparent")
        rb_transparent.pack(pady=5, anchor="w", padx=20)

        rb_green = ctk.CTkRadioButton(control_panel, text="Virtueller Greenscreen", variable=self.bg_mode,
                                      value="Green")
        rb_green.pack(pady=5, anchor="w", padx=20)

        rb_custom = ctk.CTkRadioButton(control_panel, text="Eigenes Hintergrundbild", variable=self.bg_mode,
                                       value="CustomImage")
        rb_custom.pack(pady=5, anchor="w", padx=20)

        self.btn_load_bg = ctk.CTkButton(control_panel, text="Finder öffnen & Bild laden",
                                         command=self.trigger_background_load, state="disabled")
        self.btn_load_bg.pack(pady=(5, 10), padx=20, fill="x")

        ctk.CTkLabel(control_panel, text="Kanten-Schrumpfung", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(25, 0))
        self.slider_erode = ctk.CTkSlider(control_panel, from_=0, to=10, number_of_steps=10, variable=self.edge_erode)
        self.slider_erode.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Kanten-Weichheit", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(15, 0))
        self.slider_soft = ctk.CTkSlider(control_panel, from_=1, to=21, number_of_steps=10, variable=self.edge_soft)
        self.slider_soft.pack(pady=5, padx=10, fill="x")

        self.bg_mode.trace_add("write", self.update_bg_button_state)

    def _on_preview_frame_configure(self, event):
        margin = 12
        available_w = max(120, int(event.width) - margin * 2)
        available_h = max(80, int(event.height) - margin * 2)
        available_w = min(available_w, self.ui_w)
        available_h = min(available_h, self.ui_h)
        aspect = self.ui_w / max(1, self.ui_h)
        display_w = available_w
        display_h = int(display_w / aspect)
        if display_h > available_h:
            display_h = available_h
            display_w = int(display_h * aspect)
        self.preview_display_w = max(120, display_w)
        self.preview_display_h = max(80, display_h)
        if hasattr(self, "video_label"):
            self.video_label.configure(width=self.preview_display_w, height=self.preview_display_h)

    def _start_control_panel_resize(self, event):
        self._control_resize_start_x = int(event.x_root)
        self._control_resize_start_width = int(self.control_panel_width)
        self.root.configure(cursor="sb_h_double_arrow")

    def _drag_control_panel_resize(self, event):
        delta = self._control_resize_start_x - int(event.x_root)
        new_width = self._control_resize_start_width + delta
        root_w = max(600, int(self.root.winfo_width()))
        dynamic_max = min(self.control_panel_max_width, max(self.control_panel_min_width, root_w - 280))
        self.control_panel_width = int(max(self.control_panel_min_width, min(dynamic_max, new_width)))
        if hasattr(self, "control_shell"):
            self.control_shell.configure(width=self.control_panel_width)
        if hasattr(self, "control_panel"):
            self.control_panel.configure(width=max(220, self.control_panel_width - 26))
        self._update_status_wraplengths()

    def _stop_control_panel_resize(self, _event):
        self.root.configure(cursor="")

    def _update_status_wraplengths(self):
        wrap = max(180, self.control_panel_width - 48)
        for widget_name in (
            "model_status_label",
            "yolo_status_label",
            "corridor_status_label",
            "live_output_status_label",
            "post_input_label",
            "post_output_label",
            "post_status_label",
            "metrics_summary_label",
            "metrics_label",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.configure(wraplength=wrap)
                except Exception:
                    pass

    def _add_metrics_section(self, control_panel):
        self.metrics_header = ctk.CTkFrame(control_panel, fg_color="transparent")
        self.metrics_header.pack(pady=(10, 5), padx=10, fill="x")
        ctk.CTkLabel(self.metrics_header, text="Status / Metriken", font=ctk.CTkFont(size=14, weight="bold")).pack(
            side="left")
        self.btn_metrics_toggle = ctk.CTkButton(
            self.metrics_header,
            text="-",
            width=26,
            height=24,
            command=self.toggle_metrics_panel
        )
        self.btn_metrics_toggle.pack(side="right", padx=(4, 0))
        self.btn_metrics_info = ctk.CTkButton(
            self.metrics_header,
            text="i",
            width=26,
            height=24,
            command=self.show_metrics_info
        )
        self.btn_metrics_info.pack(side="right")

        self.metrics_panel = ctk.CTkFrame(control_panel, fg_color="#242424", corner_radius=6)
        self.metrics_panel.pack(pady=(0, 10), padx=10, fill="x")
        self.metrics_summary_label = ctk.CTkLabel(
            self.metrics_panel,
            text="Wartet auf Live-Daten",
            justify="left",
            anchor="w",
            wraplength=300,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.metrics_summary_label.pack(pady=(8, 2), padx=10, fill="x")
        self.metrics_label = ctk.CTkLabel(
            self.metrics_panel,
            text=self.metrics_text,
            font=ctk.CTkFont(family="Consolas", size=12),
            justify="left",
            anchor="w",
            wraplength=300
        )
        self.metrics_label.pack(pady=(0, 8), padx=10, fill="x")

    def toggle_metrics_panel(self):
        if self.metrics_expanded.get():
            self.metrics_expanded.set(False)
            self.metrics_label.pack_forget()
            self.btn_metrics_toggle.configure(text="+")
        else:
            self.metrics_expanded.set(True)
            self.metrics_label.pack(pady=(0, 8), padx=10, fill="x")
            self.btn_metrics_toggle.configure(text="-")

    def _metrics_summary_text(self, metrics_text):
        lines = [line.strip() for line in str(metrics_text or "").splitlines() if line.strip()]
        if not lines:
            return "Keine Metriken"
        if lines[0] == "Pipeline Live":
            ai_line = next((line for line in lines if line.startswith("AI/Fertig/SDI:")), "")
            drop_line = next((line for line in lines if line.startswith("Drops:")), "")
            if ai_line and drop_line:
                return ai_line + "\n" + drop_line
        if lines[0] == "Performance":
            fps_line = next((line for line in lines if line.startswith("Verarbeitet:")), "")
            drop_line = next((line for line in lines if line.startswith("Verworfen:")), "")
            if fps_line and drop_line:
                return fps_line + "\n" + drop_line
        return lines[0]

    def setup_gui(self):
        self.preview_frame = ctk.CTkFrame(self.root, fg_color="#1f1f1f")
        self.preview_frame.pack(side="left", padx=(20, 8), pady=20, expand=True, fill="both")
        self.preview_frame.bind("<Configure>", self._on_preview_frame_configure)
        self.video_label = ctk.CTkLabel(
            self.preview_frame,
            text="Kamera gestoppt",
            width=self.ui_w,
            height=self.ui_h,
            fg_color="#2b2b2b",
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.control_shell = ctk.CTkFrame(self.root, width=self.control_panel_width, fg_color="transparent")
        self.control_shell.pack(side="right", fill="y", padx=(0, 20), pady=20)
        self.control_shell.pack_propagate(False)
        self.control_shell.grid_propagate(False)
        self.control_shell.grid_rowconfigure(0, weight=1)
        self.control_shell.grid_columnconfigure(1, weight=1)
        self.control_resize_handle = ctk.CTkFrame(self.control_shell, width=10, fg_color="#242424", corner_radius=4)
        self.control_resize_handle.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.control_resize_handle.bind("<Enter>", lambda _event: self.root.configure(cursor="sb_h_double_arrow"))
        self.control_resize_handle.bind("<Leave>", lambda _event: self.root.configure(cursor=""))
        self.control_resize_handle.bind("<ButtonPress-1>", self._start_control_panel_resize)
        self.control_resize_handle.bind("<B1-Motion>", self._drag_control_panel_resize)
        self.control_resize_handle.bind("<ButtonRelease-1>", self._stop_control_panel_resize)

        control_panel = ctk.CTkScrollableFrame(self.control_shell, width=max(220, self.control_panel_width - 26))
        control_panel.grid(row=0, column=1, sticky="nsew")
        self.control_panel = control_panel

        title = ctk.CTkLabel(control_panel, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=15)

        ctk.CTkLabel(control_panel, text="AI-Modell", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5, 5))
        self.model_select = ctk.CTkOptionMenu(control_panel, values=MODEL_OPTIONS,
                                              variable=self.model_name,
                                              command=self.change_model)
        self.model_select.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Main AI Hardware", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.main_ai_device_select = ctk.CTkOptionMenu(
            control_panel,
            values=MAIN_AI_DEVICE_OPTIONS,
            variable=self.main_ai_device_mode,
            command=self.change_main_ai_device
        )
        self.main_ai_device_select.pack(pady=(3, 6), padx=10, fill="x")

        self.model_status_label = ctk.CTkLabel(control_panel, text=self.model_status, wraplength=300,
                                               justify="left")
        self.model_status_label.pack(pady=(0, 10), padx=10, fill="x")

        self._add_metrics_section(control_panel)

        ctk.CTkLabel(control_panel, text="Fensteransicht", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(10, 5))
        self.view_switch = ctk.CTkSegmentedButton(
            control_panel,
            values=["Input", "Alpha Matte", "Processed"],
            variable=self.view_mode
        )
        self.view_switch.pack(pady=(0, 10), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Hintergrund-Effekt", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(10, 5))

        rb_checker = ctk.CTkRadioButton(control_panel, text="Karo-Muster (Transparenz)", variable=self.bg_mode,
                                        value="Checker")
        rb_checker.pack(pady=5, anchor="w", padx=20)

        rb_transparent = ctk.CTkRadioButton(control_panel, text="Echt Transparent", variable=self.bg_mode,
                                            value="Transparent")
        rb_transparent.pack(pady=5, anchor="w", padx=20)

        rb_green = ctk.CTkRadioButton(control_panel, text="Virtueller Greenscreen", variable=self.bg_mode,
                                      value="Green")
        rb_green.pack(pady=5, anchor="w", padx=20)

        rb_custom = ctk.CTkRadioButton(control_panel, text="Eigenes Hintergrundbild", variable=self.bg_mode,
                                       value="CustomImage")
        rb_custom.pack(pady=5, anchor="w", padx=20)

        self.btn_load_bg = ctk.CTkButton(control_panel, text="Finder öffnen & Bild laden",
                                         command=self.trigger_background_load, state="disabled")
        self.btn_load_bg.pack(pady=(5, 10), padx=20, fill="x")

        ctk.CTkLabel(control_panel, text="YOLO Objektauswahl", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(12, 5))
        self.chk_yolo_enabled = ctk.CTkCheckBox(
            control_panel,
            text="YOLO-Nachbearbeitung aktiv",
            variable=self.yolo_enabled,
            command=self.toggle_yolo
        )
        self.chk_yolo_enabled.pack(pady=(0, 6), padx=20, anchor="w")
        self.yolo_model_select = ctk.CTkOptionMenu(
            control_panel,
            values=YOLO_MODEL_OPTIONS,
            variable=self.yolo_model_name,
            command=self.change_yolo_model
        )
        self.yolo_model_select.pack(pady=(0, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="YOLO Hardware", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.yolo_device_select = ctk.CTkOptionMenu(
            control_panel,
            values=YOLO_DEVICE_OPTIONS,
            variable=self.yolo_device_mode,
            command=self.change_yolo_device
        )
        self.yolo_device_select.pack(pady=(3, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="YOLO Confidence", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.yolo_conf_slider = ctk.CTkSlider(
            control_panel,
            from_=0.1,
            to=0.8,
            number_of_steps=14,
            variable=self.yolo_confidence
        )
        self.yolo_conf_slider.pack(pady=(3, 6), padx=10, fill="x")
        self.chk_yolo_sync_postprocess = ctk.CTkCheckBox(
            control_panel,
            text="YOLO synchron nach Main AI",
            variable=self.yolo_sync_postprocess
        )
        self.chk_yolo_sync_postprocess.pack(pady=(0, 6), padx=20, anchor="w")
        self.chk_yolo_select_all = ctk.CTkCheckBox(
            control_panel,
            text="Neue Objekte automatisch auswählen",
            variable=self.yolo_select_all
        )
        self.chk_yolo_select_all.pack(pady=(0, 6), padx=20, anchor="w")
        self.yolo_status_label = ctk.CTkLabel(
            control_panel,
            textvariable=self.yolo_status,
            wraplength=300,
            justify="left"
        )
        self.yolo_status_label.pack(pady=(0, 6), padx=10, fill="x")
        self.yolo_objects_frame = ctk.CTkFrame(control_panel, fg_color="transparent")

        self.corridor_header = ctk.CTkLabel(
            control_panel,
            text="CorridorKey Greenscreen",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.corridor_header.pack(
            pady=(12, 5))
        self.chk_corridor_enabled = ctk.CTkCheckBox(
            control_panel,
            text="CorridorKey-Refinement aktiv",
            variable=self.corridor_enabled,
            command=self.toggle_corridor_key
        )
        self.chk_corridor_enabled.pack(pady=(0, 6), padx=20, anchor="w")
        ctk.CTkLabel(control_panel, text="CorridorKey Hardware", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.corridor_device_select = ctk.CTkOptionMenu(
            control_panel,
            values=CORRIDORKEY_DEVICE_OPTIONS,
            variable=self.corridor_device_mode,
            command=self.change_corridor_device
        )
        self.corridor_device_select.pack(pady=(3, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="CorridorKey Despill", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.corridor_despill_slider = ctk.CTkSlider(
            control_panel,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=self.corridor_despill_strength,
            command=lambda _value: self._update_corridor_status_settings()
        )
        self.corridor_despill_slider.pack(pady=(3, 6), padx=10, fill="x")
        ctk.CTkLabel(control_panel, text="CorridorKey Despeckle", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(4, 0))
        self.corridor_despeckle_slider = ctk.CTkSlider(
            control_panel,
            from_=0,
            to=1200,
            number_of_steps=24,
            variable=self.corridor_despeckle_size,
            command=lambda _value: self._update_corridor_status_settings()
        )
        self.corridor_despeckle_slider.pack(pady=(3, 6), padx=10, fill="x")
        self.corridor_status_label = ctk.CTkLabel(
            control_panel,
            textvariable=self.corridor_status,
            wraplength=300,
            justify="left"
        )
        self.corridor_status_label.pack(pady=(0, 6), padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Optimierung", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(12, 5))
        ctk.CTkLabel(control_panel, text="Kanten-Schrumpfung", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(8, 0))
        self.slider_erode = ctk.CTkSlider(control_panel, from_=0, to=10, number_of_steps=10, variable=self.edge_erode)
        self.slider_erode.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(control_panel, text="Kanten-Weichheit", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(10, 0))
        self.slider_soft = ctk.CTkSlider(control_panel, from_=1, to=21, number_of_steps=10, variable=self.edge_soft)
        self.slider_soft.pack(pady=5, padx=10, fill="x")
        self.chk_fast_live_alpha = ctk.CTkCheckBox(
            control_panel,
            text="Live Fast Alpha",
            variable=self.fast_live_alpha,
            command=self.toggle_fast_live_alpha
        )
        self.chk_fast_live_alpha.pack(pady=(0, 6), padx=20, anchor="w")

        ctk.CTkLabel(control_panel, text="Arbeitsmodus", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(12, 5))
        self.mode_switch = ctk.CTkSegmentedButton(
            control_panel,
            values=["Live", "Postproduktion"],
            variable=self.app_mode,
            command=self.change_app_mode
        )
        self.mode_switch.pack(pady=(0, 10), padx=10, fill="x")

        self.live_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        live_sources = run_with_timeout(get_live_input_sources, ["Keine Live-Quelle gefunden"], timeout=8.0)
        self.current_live_source = live_sources[0]
        self.camera_select = ctk.CTkOptionMenu(self.live_frame, values=live_sources,
                                               command=self.change_camera)
        self.camera_select.set(self.current_live_source)
        self.camera_select.pack(pady=10, padx=10, fill="x")

        self.btn_refresh_cameras = ctk.CTkButton(self.live_frame, text="Kameras neu suchen",
                                                command=self.refresh_cameras)
        self.btn_refresh_cameras.pack(pady=5, padx=10, fill="x")

        self.btn_toggle = ctk.CTkButton(self.live_frame, text="Kamera Starten", command=self.toggle_camera)
        self.btn_toggle.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(self.live_frame, text="Live Output DeckLink", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(14, 5), padx=10, anchor="w")
        self.decklink_device_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=5.0),
            variable=self.live_output_device,
            command=lambda _choice: self.restart_live_output_if_needed()
        )
        self.decklink_device_select.pack(pady=(4, 4), padx=10, fill="x")
        ctk.CTkLabel(self.live_frame, text="Alpha Matte SDI", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(8, 0), padx=10, anchor="w")
        self.decklink_key_device_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=5.0),
            variable=self.live_key_output_device,
            command=lambda _choice: self.restart_live_output_if_needed()
        )
        self.decklink_key_device_select.pack(pady=(4, 4), padx=10, fill="x")
        self.decklink_mode_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=list(DECKLINK_OUTPUT_MODES.keys()),
            variable=self.live_output_mode,
            command=lambda _choice: self.restart_decklink_io_if_needed()
        )
        self.decklink_mode_select.pack(pady=4, padx=10, fill="x")
        self.btn_refresh_decklink = ctk.CTkButton(
            self.live_frame,
            text="DeckLink Geräte neu suchen",
            command=self.refresh_decklink_devices
        )
        self.btn_refresh_decklink.pack(pady=4, padx=10, fill="x")
        self.chk_live_output = ctk.CTkCheckBox(
            self.live_frame,
            text="Live Output aktiv",
            variable=self.live_output_enabled,
            command=self.toggle_live_output
        )
        self.chk_live_output.pack(pady=(8, 4), padx=20, anchor="w")
        self.chk_live_key_output = ctk.CTkCheckBox(
            self.live_frame,
            text="Alpha Matte auf zweitem SDI",
            variable=self.live_key_output_enabled,
            command=self.toggle_live_output
        )
        self.chk_live_key_output.pack(pady=(2, 4), padx=20, anchor="w")
        ctk.CTkLabel(self.live_frame, text="Output Sync", font=ctk.CTkFont(size=12, weight="bold")).pack(
            pady=(8, 0), padx=10, anchor="w")
        self.sync_overlay_select = ctk.CTkOptionMenu(
            self.live_frame,
            values=["Aus", "Frame", "Timecode", "Beides"],
            variable=self.sync_overlay_mode
        )
        self.sync_overlay_select.pack(pady=(4, 4), padx=10, fill="x")
        fill_sync_row = ctk.CTkFrame(self.live_frame, fg_color="transparent")
        fill_sync_row.pack(pady=(2, 2), padx=10, fill="x")
        ctk.CTkLabel(fill_sync_row, text="Fill Delay", width=90, anchor="w").pack(side="left")
        ctk.CTkButton(fill_sync_row, text="-", width=32, command=lambda: self.adjust_output_delay("fill", -1)).pack(side="left", padx=2)
        self.fill_delay_label = ctk.CTkLabel(fill_sync_row, textvariable=self.fill_delay_frames, width=34)
        self.fill_delay_label.pack(side="left", padx=4)
        ctk.CTkButton(fill_sync_row, text="+", width=32, command=lambda: self.adjust_output_delay("fill", 1)).pack(side="left", padx=2)
        matte_sync_row = ctk.CTkFrame(self.live_frame, fg_color="transparent")
        matte_sync_row.pack(pady=(2, 6), padx=10, fill="x")
        ctk.CTkLabel(matte_sync_row, text="Matte Delay", width=90, anchor="w").pack(side="left")
        ctk.CTkButton(matte_sync_row, text="-", width=32, command=lambda: self.adjust_output_delay("matte", -1)).pack(side="left", padx=2)
        self.matte_delay_label = ctk.CTkLabel(matte_sync_row, textvariable=self.matte_delay_frames, width=34)
        self.matte_delay_label.pack(side="left", padx=4)
        ctk.CTkButton(matte_sync_row, text="+", width=32, command=lambda: self.adjust_output_delay("matte", 1)).pack(side="left", padx=2)
        self.live_output_status_label = ctk.CTkLabel(
            self.live_frame,
            textvariable=self.live_output_status,
            wraplength=300,
            justify="left"
        )
        self.live_output_status_label.pack(pady=(0, 8), padx=10, fill="x")

        self.post_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        self.btn_post_input = ctk.CTkButton(self.post_frame, text="Quelldatei wählen", command=self.select_post_input)
        self.btn_post_input.pack(pady=(8, 4), padx=10, fill="x")
        self.post_input_label = ctk.CTkLabel(self.post_frame, textvariable=self.post_input_path,
                                             wraplength=300, justify="left")
        self.post_input_label.pack(pady=(0, 8), padx=10, fill="x")

        self.btn_post_output = ctk.CTkButton(self.post_frame, text="Speicherziel wählen", command=self.select_post_output)
        self.btn_post_output.pack(pady=4, padx=10, fill="x")
        self.post_output_label = ctk.CTkLabel(self.post_frame, textvariable=self.post_output_path,
                                              wraplength=300, justify="left")
        self.post_output_label.pack(pady=(0, 8), padx=10, fill="x")

        self.post_progress = ctk.CTkProgressBar(self.post_frame)
        self.post_progress.set(0)
        self.post_progress.pack(pady=(6, 4), padx=10, fill="x")
        self.post_status_label = ctk.CTkLabel(self.post_frame, textvariable=self.post_status,
                                              wraplength=300, justify="left")
        self.post_status_label.pack(pady=(0, 8), padx=10, fill="x")

        self.btn_post_process = ctk.CTkButton(self.post_frame, text="Datei verarbeiten",
                                              command=self.start_post_processing)
        self.btn_post_process.pack(pady=(4, 12), padx=10, fill="x")

        self.bg_mode.trace_add("write", self.update_bg_button_state)
        self.view_mode.trace_add("write", self.refresh_display_view)
        self._update_status_wraplengths()
        self.change_app_mode(self.app_mode.get())
