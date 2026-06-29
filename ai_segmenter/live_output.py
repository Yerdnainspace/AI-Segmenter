import threading

import cv2

from ai_segmenter.config import DECKLINK_OUTPUT_MODES
from ai_segmenter.decklink import DeckLinkLiveOutput, get_decklink_input_devices, get_decklink_output_devices
from ai_segmenter.utils import run_with_timeout


class LiveOutputMixin:
    def refresh_decklink_devices(self):
        if getattr(self, "_decklink_refresh_running", False):
            return
        self._decklink_refresh_running = True
        try:
            self.btn_refresh_decklink.configure(text="Suche laeuft...", state="disabled")
        except Exception:
            pass
        self.live_output_status.set("DeckLink Geraete werden gesucht ...")

        def worker():
            try:
                values = run_with_timeout(get_decklink_output_devices, ["Keine DeckLink-Ausgabe gefunden"], timeout=6.0)
                inputs = run_with_timeout(get_decklink_input_devices, [], timeout=6.0)
                error = None
            except Exception as exc:
                values = None
                inputs = []
                error = exc
            self.root.after(0, lambda: self._finish_decklink_refresh(values, inputs, error))

        threading.Thread(target=worker, name="DeckLinkRefresh", daemon=True).start()

    def _finish_decklink_refresh(self, values, inputs=None, error=None):
        self._decklink_refresh_running = False
        try:
            self.btn_refresh_decklink.configure(text="DeckLink Geraete neu suchen", state="normal")
        except Exception:
            pass
        if error is not None:
            self.live_output_status.set(f"DeckLink Suche Fehler: {error}")
            return
        if not values:
            values = ["Keine DeckLink-Ausgabe gefunden"]
        self.decklink_device_select.configure(values=values)
        self.decklink_key_device_select.configure(values=values)
        if self.live_output_device.get() not in values:
            self.live_output_device.set(values[0])
        if self.live_key_output_device.get() not in values:
            self.live_key_output_device.set(values[1] if len(values) > 1 else values[0])
        existing_sources = []
        try:
            existing_sources = list(self.camera_select.cget("values") or [])
        except Exception:
            existing_sources = []
        non_decklink_sources = [
            source for source in existing_sources
            if source and not str(source).startswith("DeckLink: ") and source != "Keine Live-Quelle gefunden"
        ]
        live_sources = [*non_decklink_sources, *[f"DeckLink: {name}" for name in (inputs or [])]]
        if not live_sources:
            live_sources = ["Keine Live-Quelle gefunden"]
        self.camera_select.configure(values=live_sources)
        if self.current_live_source not in live_sources:
            self.current_live_source = live_sources[0]
            self.camera_select.set(self.current_live_source)
        self.live_output_status.set("DeckLink Geraeteliste aktualisiert.")

    def toggle_live_output(self):
        if self.live_output_enabled.get() or self.live_key_output_enabled.get():
            self.start_live_output()
        else:
            self.stop_live_output()

    def restart_live_output_if_needed(self):
        if self.live_output_enabled.get() or self.live_key_output_enabled.get():
            self.stop_live_output()
            self.start_live_output()

    def restart_decklink_io_if_needed(self):
        self.restart_live_output_if_needed()
        if self.is_running and (self.current_live_source or "").startswith("DeckLink: "):
            self._stop_camera_internal(preserve_preview=True)
            self.root.after(150, lambda: self._start_camera_internal(preserve_preview=True))

    def adjust_output_delay(self, target, delta):
        if target == "fill":
            self.fill_delay_frames.set(max(0, min(120, int(self.fill_delay_frames.get()) + int(delta))))
            self.fill_delay_buffer.clear()
        elif target == "matte":
            self.matte_delay_frames.set(max(0, min(120, int(self.matte_delay_frames.get()) + int(delta))))
            self.matte_delay_buffer.clear()
        self._update_live_output_sync_status()

    def _reset_output_sync_buffers(self):
        self.live_output_frame_counter = 0
        self.fill_delay_buffer.clear()
        self.matte_delay_buffer.clear()

    def _update_live_output_sync_status(self):
        base = self.live_output_status.get()
        lines = [line for line in base.splitlines() if not line.startswith("Sync:")]
        lines.append(
            f"Sync: Overlay {self.sync_overlay_mode.get()} | "
            f"Fill +{int(self.fill_delay_frames.get())}F | Matte +{int(self.matte_delay_frames.get())}F"
        )
        self.live_output_status.set("\n".join(lines))

    def _current_output_fps(self):
        mode = self.live_output_mode.get()
        if mode in DECKLINK_OUTPUT_MODES:
            return float(DECKLINK_OUTPUT_MODES[mode][3])
        return 25.0

    def _format_output_timecode(self, frame_index):
        fps = max(1, int(round(self._current_output_fps())))
        frame_index = max(0, int(frame_index))
        frames = frame_index % fps
        total_seconds = frame_index // fps
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _sync_overlay_text(self, frame_index, stream_label):
        mode = self.sync_overlay_mode.get()
        if mode == "Aus":
            return None
        parts = [stream_label]
        if mode in ("Frame", "Beides"):
            parts.append(f"F{int(frame_index):06d}")
        if mode in ("Timecode", "Beides"):
            parts.append(self._format_output_timecode(frame_index))
        if self.yolo_enabled.get() and not self._is_yolo_primary_model():
            parts.append("YOLO-S" if self.yolo_sync_postprocess.get() else "YOLO-A")
        return "  ".join(parts)

    def _draw_output_sync_overlay(self, frame_rgb, frame_index, stream_label):
        text = self._sync_overlay_text(frame_index, stream_label)
        if not text:
            return frame_rgb
        out = frame_rgb.copy()
        h, w = out.shape[:2]
        scale = max(0.7, min(2.2, w / 960.0))
        thickness = max(2, int(round(scale * 2)))
        margin = max(14, int(18 * scale))
        y = h - margin
        x = margin
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            out,
            (x - 8, y - th - baseline - 8),
            (x + tw + 8, y + baseline + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out

    def _delay_output_frame(self, buffer, frame, delay_frames):
        delay_frames = max(0, int(delay_frames))
        if delay_frames <= 0:
            buffer.clear()
            return frame
        buffer.append(frame.copy())
        if len(buffer) <= delay_frames:
            return buffer[0]
        return buffer.popleft()

    def start_live_output(self):
        self.stop_live_output()
        self._reset_output_sync_buffers()

        device_name = self.live_output_device.get()
        if device_name == "Keine DeckLink-Ausgabe gefunden":
            self.live_output_enabled.set(False)
            self.live_output_status.set("Keine DeckLink-Ausgabe gefunden.")
            return

        mode_label = self.live_output_mode.get()
        if mode_label not in DECKLINK_OUTPUT_MODES:
            self.live_output_enabled.set(False)
            self.live_output_status.set("Ungueltiger DeckLink-Modus.")
            return

        try:
            started = []
            if self.live_output_enabled.get():
                self.decklink_output = DeckLinkLiveOutput(
                    device_name,
                    mode_label,
                    status_callback=lambda text: self.root.after(0, lambda: self.live_output_status.set(text))
                )
                self.decklink_output.start()
                started.append(f"Fill: {device_name}")

            if self.live_key_output_enabled.get():
                key_device_name = self.live_key_output_device.get()
                if key_device_name == "Keine DeckLink-Ausgabe gefunden":
                    raise RuntimeError("Keine zweite DeckLink-Ausgabe fuer Alpha Matte gefunden.")
                if key_device_name == device_name and self.live_output_enabled.get():
                    raise RuntimeError("Fill und Alpha Matte muessen auf verschiedene DeckLink-Ausgaenge gelegt werden.")
                self.decklink_key_output = DeckLinkLiveOutput(
                    key_device_name,
                    mode_label,
                    status_callback=None
                )
                self.decklink_key_output.start()
                started.append(f"Key/Matte: {key_device_name}")

            if started:
                self.live_output_status.set("DeckLink aktiv:\n" + "\n".join(started) + f"\n{mode_label}")
                self._update_live_output_sync_status()
        except Exception as exc:
            if self.decklink_output is not None:
                self.decklink_output.stop()
            if self.decklink_key_output is not None:
                self.decklink_key_output.stop()
            self.decklink_output = None
            self.decklink_key_output = None
            self.live_output_enabled.set(False)
            self.live_key_output_enabled.set(False)
            self.live_output_status.set(f"DeckLink Startfehler: {exc}")

    def stop_live_output(self):
        self._reset_output_sync_buffers()
        if self.decklink_output is not None:
            self.decklink_output.stop()
            self.decklink_output = None
        if self.decklink_key_output is not None:
            self.decklink_key_output.stop()
            self.decklink_key_output = None
        if not self.live_output_enabled.get() and not self.live_key_output_enabled.get():
            self.live_output_status.set("DeckLink Output aus")

    def write_live_output_frame(self, rgb_frame, alpha_2d, processed_frame):
        if self.decklink_output is None and self.decklink_key_output is None:
            return False
        wrote_frame = False
        try:
            frame_index = self.live_output_frame_counter
            self.live_output_frame_counter += 1
            if self.decklink_output is not None:
                fill_frame = self._make_view_frame(rgb_frame, alpha_2d, processed_frame)
                if fill_frame.shape[2] == 4:
                    fill_frame = fill_frame[:, :, :3]
                fill_frame = self._draw_output_sync_overlay(fill_frame, frame_index, "FILL")
                fill_frame = self._delay_output_frame(
                    self.fill_delay_buffer,
                    fill_frame,
                    self.fill_delay_frames.get()
                )
                self.decklink_output.write(fill_frame)
                wrote_frame = True
            if self.decklink_key_output is not None:
                alpha_u8 = self._alpha_to_u8(alpha_2d)
                alpha_rgb = cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB)
                alpha_rgb = self._draw_output_sync_overlay(alpha_rgb, frame_index, "MATTE")
                alpha_rgb = self._delay_output_frame(
                    self.matte_delay_buffer,
                    alpha_rgb,
                    self.matte_delay_frames.get()
                )
                self.decklink_key_output.write(alpha_rgb)
                wrote_frame = True
        except Exception as exc:
            self.live_output_status.set(f"DeckLink Schreibfehler: {exc}")
        return wrote_frame



