import os
import time

import customtkinter as ctk

from ai_segmenter.profiler import PipelineProfiler


class MetricsMixin:
    def show_metrics_info(self):
        info = ctk.CTkToplevel(self.root)
        info.title("Live-Metriken")
        info.geometry("520x560")
        info.transient(self.root)
        info.grab_set()

        text = (
            "Bedeutung der Messwerte\n\n"
            "Modell: Aktuell geladener Segmentierer.\n\n"
            "Backend: OpenCV-Kamera-Backend, z.B. DirectShow oder Media Foundation.\n\n"
            "Quelle: Von der Kamera gelieferte Aufloesung und gemessene Eingangs-FPS. "
            "Der OpenCV-FPS-Wert wird nicht blind verwendet, weil Capture-Hardware wie "
            "BMD/Blackmagic haeufig 30 FPS meldet, obwohl das Signal z.B. 25 FPS hat.\n\n"
            "Verarbeitet: Bilder pro Sekunde, die komplett durch Kamera, Modell, "
            "Postprocessing und Anzeige-Pipeline laufen.\n\n"
            "Verworfen: Geschaetzte Differenz zwischen Quell-FPS und verarbeiteten FPS. "
            "OpenCV liefert meist keine echten Drop-Frame-Zaehler, daher ist das ein "
            "Vergleichswert fuer Modell-Benchmarks.\n\n"
            "Latenz gesamt: Zeit ab erfolgreich gelesenem Kamerabild bis fertigem Ausgabebild. "
            "Die Wartezeit auf das naechste Kamerabild wird nicht mitgezaehlt.\n\n"
            "Rechenzeit: Preprocess, Main AI, Alpha/Postprocessing und Compose pro verarbeitetem Frame. "
            "Daraus wird die theoretische Max-FPS berechnet, bevor diese Pipeline-Stufe Frames verwerfen muss.\n\n"
            "Capture bis naechstes Frame: blockierende Wartezeit bis das naechste Kamerabild "
            "verfuegbar ist. Diese Zeit wird getrennt von der Rechenzeit angezeigt.\n\n"
            "Main AI: Reine Modellzeit fuer die Haupt-Maskenberechnung, z.B. RVM, "
            "BiRefNet, MediaPipe oder YOLO als Hauptmodell.\n\n"

            "YOLO sync: YOLO-Nachbearbeitung auf demselben Frame nach der Main AI. "
            "Diese Zeit ist framegenau und Teil der Gesamt-Latenz.\n\n"
            "YOLO async: YOLO-Nachbearbeitung im Neben-Thread. Diese Zeit wird nicht zur "
            "Frame-Latenz addiert, weil die Hauptpipeline mit der zuletzt verfuegbaren "
            "Objektauswahl weiterlaeuft. Das Alter zeigt, wie alt die aktuelle YOLO-Erkennung ist.\n\n"
            "Eingehend: Geschaetzte Rohdatenrate der Kamera auf Basis von "
            "Aufloesung x 3 Farbkanalen x Quell-FPS.\n\n"
            "Verarbeitet: Geschaetzte Rohdatenrate der Frames, die die Pipeline wirklich "
            "fertig verarbeitet.\n\n"
            "Frames: Anzahl der seit Kamerastart fertig verarbeiteten Frames.\n\n"
            "Lesefehler: Fehlgeschlagene Kamera-Reads."
        )

        label = ctk.CTkLabel(info, text=text, justify="left", anchor="nw", wraplength=470)
        label.pack(padx=20, pady=20, fill="both", expand=True)
        ctk.CTkButton(info, text="Schliessen", command=info.destroy).pack(pady=(0, 16))

    def _reset_perf_metrics(self, source_fps=0.0, backend_name="-", capture_size=(0, 0)):
        now = time.perf_counter()
        self._perf_window_start = now
        self._perf_total_frames = 0
        self._perf_window_frames = 0
        self._perf_read_failures = 0
        self._perf_source_fps = float(source_fps or 0.0)
        self._perf_backend_name = backend_name or "-"
        self._perf_capture_size = capture_size
        self._perf_last_text_update = 0.0
        self._perf_ema = {
            "total_ms": 0.0,
            "compute_ms": 0.0,
            "preprocess_ms": 0.0,
            "infer_ms": 0.0,
            "post_ms": 0.0,
            "compose_ms": 0.0,
        }
        with self.metrics_lock:
            if self.is_running:
                self.metrics_text = "Performance\nMesse Daten ..."
            else:
                self.metrics_text = "Performance\nKamera gestoppt"

    def _start_pipeline_profiler(self, source, backend_name, capture_size, source_fps):
        self._stop_pipeline_profiler()
        with self.model_lock:
            segmenter = self.segmenter
        model_device = getattr(segmenter, "device_label", "")
        model_dtype = str(getattr(segmenter, "model_dtype", ""))
        model_backend = getattr(segmenter, "tensorrt_status", "")
        profiler = PipelineProfiler()
        path = profiler.start(
            {
                "source": source,
                "model": self.loaded_model_name,
                "model_device": model_device,
                "model_dtype": model_dtype,
                "model_backend": model_backend,
                "backend": backend_name,
                "capture_width": int(capture_size[0]),
                "capture_height": int(capture_size[1]),
                "source_fps": float(source_fps or 0.0),
            }
        )
        self.pipeline_profiler = profiler
        self.pipeline_log_path = path
        return profiler

    def _stop_pipeline_profiler(self):
        profiler = self.pipeline_profiler
        self.pipeline_profiler = None
        if profiler is not None:
            profiler.close()

    def _update_perf_metrics(self, frame_shape, total_ms, compute_ms, preprocess_ms, infer_ms, post_ms, compose_ms):
        now = time.perf_counter()
        self._perf_total_frames += 1
        self._perf_window_frames += 1

        alpha = 0.18
        values = {
            "total_ms": total_ms,
            "compute_ms": compute_ms,
            "preprocess_ms": preprocess_ms,
            "infer_ms": infer_ms,
            "post_ms": post_ms,
            "compose_ms": compose_ms,
        }
        for key, value in values.items():
            previous = self._perf_ema.get(key, 0.0)
            self._perf_ema[key] = value if previous <= 0.0 else (previous * (1.0 - alpha) + value * alpha)

        elapsed = now - self._perf_window_start
        if elapsed < 0.5:
            return

        processed_fps = self._perf_window_frames / elapsed if elapsed > 0 else 0.0
        source_fps = self._perf_source_fps if self._perf_source_fps > 1.0 else 0.0
        dropped_fps = max(0.0, source_fps - processed_fps)
        dropped_percent = (dropped_fps / source_fps * 100.0) if source_fps > 0 else 0.0

        h, w = frame_shape[:2]
        bytes_per_frame = int(w * h * 3)
        incoming_mib = (bytes_per_frame * source_fps) / (1024 * 1024)
        incoming_mbit = incoming_mib * 8.0
        processed_mib = (bytes_per_frame * processed_fps) / (1024 * 1024)
        processed_mbit = processed_mib * 8.0
        max_compute_fps = 1000.0 / self._perf_ema["compute_ms"] if self._perf_ema["compute_ms"] > 0 else 0.0
        source_label = f"{source_fps:.1f} FPS gemessen" if source_fps > 0 else "unbekannt"
        dropped_label = (
            f"{dropped_fps:.1f} FPS ({dropped_percent:.0f}%)"
            if source_fps > 0
            else "unbekannt"
        )

        text = (
            "Performance\n"
            f"Modell: {self.loaded_model_name}\n"
            f"Backend: {self._perf_backend_name}\n"
            f"Quelle: {int(self._perf_capture_size[0])}x{int(self._perf_capture_size[1])} @ {source_label}\n"
            f"Verarbeitet: {processed_fps:.1f} FPS\n"
            f"Verworfen: {dropped_label}\n"
            f"Latenz gesamt: {self._perf_ema['total_ms']:.1f} ms\n"
            f"Rechenzeit: {self._perf_ema['compute_ms']:.1f} ms ({max_compute_fps:.1f} FPS max)\n"
            f"Preprocess: {self._format_duration_ms(self._perf_ema['preprocess_ms'])}\n"
            f"Main AI: {self._perf_ema['infer_ms']:.1f} ms\n"
            f"Eingehend: {incoming_mib:.1f} MiB/s ({incoming_mbit:.0f} Mbit/s)\n"
            f"Verarbeitet: {processed_mib:.1f} MiB/s ({processed_mbit:.0f} Mbit/s)\n"
            f"Frames: {self._perf_total_frames}"
        )
        if self.pipeline_log_path:
            text += f"\nLog: {os.path.basename(self.pipeline_log_path)}"
        if self._yolo_selection_active():
            age_ms = (now - self.yolo_last_analysis_time) * 1000.0 if self.yolo_last_analysis_time > 0 else 0.0
            if self.yolo_last_analysis_ms <= 0:
                text += "\nYOLO Analyse: wartet"
            elif self._is_yolo_primary_model():
                text += f"\nYOLO Analyse: {self.yolo_last_analysis_ms:.1f} ms"
            elif self.yolo_sync_postprocess.get():
                text += f"\nYOLO sync: {self.yolo_last_analysis_ms:.1f} ms"
            else:
                text += f"\nYOLO async: {self.yolo_last_analysis_ms:.1f} ms, Alter {age_ms:.0f} ms"
        if self._perf_read_failures:
            text += f"\nLesefehler: {self._perf_read_failures}"

        with self.metrics_lock:
            self.metrics_text = text

        self._perf_window_start = now
        self._perf_window_frames = 0

    def _float_summary(self, summary, key):
        try:
            return float(summary.get(key) or 0.0)
        except Exception:
            return 0.0

    def _format_duration_ms(self, value_ms):
        try:
            value_ms = float(value_ms or 0.0)
        except Exception:
            value_ms = 0.0
        if 0.0 < value_ms < 1.0:
            return f"{value_ms * 1000.0:.0f} us"
        return f"{value_ms:.1f} ms"

    def _format_pipeline_summary(self, summary):
        if not summary:
            return None
        source_fps = self._float_summary(summary, "source_arrival_fps")
        capture_fps = self._float_summary(summary, "capture_fps")
        ai_fps = self._float_summary(summary, "ai_fps")
        done_fps = self._float_summary(summary, "frame_done_fps")
        output_fps = self._float_summary(summary, "decklink_output_fps")
        ai_drop = self._float_summary(summary, "ai_input_overwrite_fps")
        output_drop = self._float_summary(summary, "output_overwrite_fps")
        source_drop = self._float_summary(summary, "source_overwrite_fps")
        read_fail = self._float_summary(summary, "read_fail_fps")
        capture_actual_ms = self._float_summary(summary, "capture_actual_avg_ms")
        capture_interval_ms = self._float_summary(summary, "capture_interval_avg_ms")
        preprocess_ms = self._float_summary(summary, "preprocess_avg_ms")
        ai_ms = self._float_summary(summary, "ai_avg_ms")
        alpha_ms = self._float_summary(summary, "alpha_post_avg_ms")
        yolo_post_ms = self._float_summary(summary, "yolo_post_avg_ms")
        corridor_ms = self._float_summary(summary, "corridor_avg_ms")
        compose_ms = self._float_summary(summary, "compose_avg_ms")
        output_ms = self._float_summary(summary, "decklink_output_avg_ms")
        total_ms = self._float_summary(summary, "total_avg_ms")
        compute_ms = self._float_summary(summary, "compute_total_avg_ms")
        capture_read_ms = self._float_summary(summary, "capture_read_avg_ms")
        max_compute_fps = 1000.0 / compute_ms if compute_ms > 0 else 0.0

        ai_drop_pct = (ai_drop / capture_fps * 100.0) if capture_fps > 0 else 0.0
        source_drop_pct = (source_drop / source_fps * 100.0) if source_fps > 0 else 0.0
        output_drop_pct = (output_drop / done_fps * 100.0) if done_fps > 0 else 0.0

        model_line = summary.get("model", self.loaded_model_name)
        model_backend = summary.get("model_backend", "")
        if model_backend:
            model_line += f" | {model_backend}"

        extra_lines = []
        if self.yolo_enabled.get() and not self._is_yolo_primary_model():
            mode = "sync" if self.yolo_sync_postprocess.get() else "async"
            yolo_ms = yolo_post_ms if self.yolo_sync_postprocess.get() else float(self.yolo_last_analysis_ms or 0.0)
            if yolo_ms > 0:
                extra = f"YOLO Nachbearbeitung ({mode}): {self._format_duration_ms(yolo_ms)}"
                if not self.yolo_sync_postprocess.get() and self.yolo_last_analysis_time > 0:
                    age_ms = (time.perf_counter() - self.yolo_last_analysis_time) * 1000.0
                    extra += f" | Alter {self._format_duration_ms(age_ms)}"
                extra_lines.append(extra)
            else:
                extra_lines.append(f"YOLO Nachbearbeitung ({mode}): wartet")
        if self.corridor_enabled.get():
            extra_lines.append(f"CorridorKey: {self._format_duration_ms(corridor_ms)}")
        optional_text = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

        return (
            "Pipeline Live\n"
            f"Modell: {model_line}\n"
            f"Quelle/Capture: {source_fps:.1f} / {capture_fps:.1f} FPS\n"
            f"AI/Fertig/SDI: {ai_fps:.1f} / {done_fps:.1f} / {output_fps:.1f} FPS\n"
            f"Drops: DeckLink {source_drop:.1f}/s ({source_drop_pct:.0f}%) | AI {ai_drop:.1f}/s ({ai_drop_pct:.0f}%) | Out {output_drop:.1f}/s ({output_drop_pct:.0f}%)\n"
            f"Zeit: Capture {self._format_duration_ms(capture_actual_ms)} | Pre {self._format_duration_ms(preprocess_ms)} | AI {self._format_duration_ms(ai_ms)} | Alpha {self._format_duration_ms(alpha_ms)} | Compose {self._format_duration_ms(compose_ms)} | SDI {self._format_duration_ms(output_ms)}"
            f"{optional_text}\n"
            f"Rechenzeit {compute_ms:.1f} ms ({max_compute_fps:.1f} FPS max) | Latenz {total_ms:.1f} ms\n"
            f"Capture bis naechstes Frame: {self._format_duration_ms(capture_interval_ms)}\n"
            f"Lesefehler: {read_fail:.2f}/s"
        )

