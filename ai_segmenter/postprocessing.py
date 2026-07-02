import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from ai_segmenter.image_utils import despill_green as _despill_green
from ai_segmenter.image_utils import ensure_odd_ksize as _ensure_odd_ksize


class PostProcessingMixin:
    def select_post_input(self):
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="Datei für Postproduktion wählen",
            filetypes=[
                ("Video/Bild", "*.mp4;*.mov;*.avi;*.mkv;*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("Videos", "*.mp4;*.mov;*.avi;*.mkv"),
                ("Bilder", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("Alle Dateien", "*.*"),
            ]
        )
        if not file_path:
            return

        self.post_input_path.set(file_path)
        if self.post_output_path.get() == "Kein Ziel gewählt":
            self.post_output_path.set(self._default_post_output_path(file_path))

    def select_post_output(self):
        from tkinter import filedialog

        input_path = self.post_input_path.get()
        initial = self._default_post_output_path(input_path) if os.path.exists(input_path) else "processed_output.mp4"
        ext = os.path.splitext(initial)[1].lower()
        filetypes = [
            ("Apple ProRes 4444 MOV", "*.mov"),
            ("MP4 Video", "*.mp4"),
            ("PNG Bild", "*.png"),
            ("JPEG Bild", "*.jpg"),
            ("Alle Dateien", "*.*"),
        ]
        file_path = filedialog.asksaveasfilename(
            title="Speicherziel wählen",
            initialfile=os.path.basename(initial),
            initialdir=os.path.dirname(initial) if os.path.dirname(initial) else None,
            defaultextension=ext if ext else ".mp4",
            filetypes=filetypes
        )
        if file_path:
            self.post_output_path.set(file_path)

    def _default_post_output_path(self, input_path):
        base, ext = os.path.splitext(input_path)
        ext = ext.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            return base + "_processed.png"
        if self.bg_mode.get() == "Transparent":
            return base + "_processed.mov"
        return base + "_processed.mp4"

    def start_post_processing(self):
        if self.post_is_processing:
            return

        input_path = self.post_input_path.get()
        output_path = self.post_output_path.get()
        if not os.path.exists(input_path):
            self.post_status.set("Bitte zuerst eine gueltige Quelldatei wählen.")
            return
        if output_path == "Kein Ziel gewählt":
            output_path = self._default_post_output_path(input_path)
            self.post_output_path.set(output_path)
        input_ext = os.path.splitext(input_path)[1].lower()
        output_ext = os.path.splitext(output_path)[1].lower()
        if self.bg_mode.get() == "Transparent":
            if input_ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and output_ext != ".png":
                output_path = os.path.splitext(output_path)[0] + ".png"
                self.post_output_path.set(output_path)
            elif input_ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and output_ext != ".mov":
                output_path = os.path.splitext(output_path)[0] + ".mov"
                self.post_output_path.set(output_path)

        self.post_is_processing = True
        self.post_progress.set(0)
        self.post_status.set("Verarbeitung startet ...")
        self.btn_post_process.configure(state="disabled")
        self.model_select.configure(state="disabled")
        threading.Thread(target=self._post_processing_worker, args=(input_path, output_path), daemon=True).start()

    def _set_post_progress(self, progress, status):
        self.post_progress.set(max(0.0, min(1.0, float(progress))))
        self.post_status.set(status)

    def _finish_post_processing(self, message, error=False):
        self.post_is_processing = False
        self.btn_post_process.configure(state="normal")
        self.model_select.configure(state="normal")
        self.post_status.set(message)
        if error:
            self.video_label.configure(image=self.empty_dummy_image, text=message)

    def _post_processing_worker(self, input_path, output_path):
        try:
            ext = os.path.splitext(input_path)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self._process_post_image(input_path, output_path)
            else:
                self._process_post_video(input_path, output_path)
            self.root.after(0, lambda: self._finish_post_processing(f"Fertig gespeichert:\n{output_path}"))
        except Exception as exc:
            self.root.after(0, lambda e=exc: self._finish_post_processing(f"Fehler: {e}", error=True))

    def _compose_processed_frame(self, rgb_frame, alpha_2d, bg_cache=None, last_bg_mode=None):
        h, w = rgb_frame.shape[:2]
        current_mode = self.bg_mode.get()

        if current_mode == "Transparent":
            alpha_u8 = self._alpha_to_u8(alpha_2d)
            output_rgba = np.dstack((rgb_frame, alpha_u8))
            self.force_bg_update = False
            return output_rgba, None, current_mode

        if current_mode != last_bg_mode or self.force_bg_update or bg_cache is None:
            if current_mode == "Checker":
                bg_cache = self._get_checker_background(w, h)
            elif current_mode == "Green":
                bg_cache = np.zeros((h, w, 3), dtype=np.uint8)
                bg_cache[:] = (0, 255, 0)
            elif current_mode == "CustomImage":
                bg_cache = self._get_custom_background(w, h)
            else:
                bg_cache = np.zeros((h, w, 3), dtype=np.uint8) + 30
            last_bg_mode = current_mode
            self.force_bg_update = False

        fg_rgb = rgb_frame
        if current_mode == "Green":
            fg_rgb = _despill_green(fg_rgb, alpha_2d)

        alpha = alpha_2d[..., np.newaxis]
        output_rgb = (fg_rgb * alpha + bg_cache * (1.0 - alpha)).astype(np.uint8)
        return output_rgb, bg_cache, last_bg_mode

    def _gpu_postprocess_compose(self, segmenter, rgb_frame, alpha_t):
        torch = segmenter.torch
        device = segmenter.device
        h, w = rgb_frame.shape[:2]
        work_dtype = getattr(segmenter, "model_dtype", torch.float16)
        if work_dtype not in (torch.float16, torch.float32):
            work_dtype = torch.float16

        if alpha_t.ndim == 3:
            alpha_t = alpha_t.squeeze()
        alpha_t = alpha_t.to(device=device, dtype=work_dtype).clamp_(0.0, 1.0)

        soft_size = _ensure_odd_ksize(int(self.edge_soft.get()), min_k=1)
        if soft_size > 1:
            pad = soft_size // 2
            alpha_t = torch.nn.functional.avg_pool2d(
                alpha_t.view(1, 1, h, w),
                kernel_size=soft_size,
                stride=1,
                padding=pad,
            ).view(h, w).clamp_(0.0, 1.0)

        erode_size = int(self.edge_erode.get())
        if erode_size > 0:
            erode_size = _ensure_odd_ksize(erode_size, min_k=1)
            gate = (alpha_t >= 0.5).to(dtype=work_dtype).view(1, 1, h, w)
            pad = erode_size // 2
            gate = -torch.nn.functional.max_pool2d(-gate, kernel_size=erode_size, stride=1, padding=pad)
            alpha_t = (alpha_t * gate.view(h, w)).clamp_(0.0, 1.0)

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_frame)).to(device=device, dtype=work_dtype)
        current_mode = self.bg_mode.get()
        alpha_u8 = (alpha_t * 255.0).clamp(0, 255).to(torch.uint8)
        if current_mode == "Transparent":
            rgb_u8 = rgb_t.clamp(0, 255).to(torch.uint8)
            output = torch.cat((rgb_u8, alpha_u8.unsqueeze(-1)), dim=2)
        else:
            if current_mode != self.pipeline_last_bg_mode or self.force_bg_update or self.pipeline_gpu_bg_cache is None:
                if current_mode == "Checker":
                    bg_np = self._get_checker_background(w, h)
                elif current_mode == "Green":
                    bg_np = np.zeros((h, w, 3), dtype=np.uint8)
                    bg_np[:] = (0, 255, 0)
                elif current_mode == "CustomImage":
                    bg_np = self._get_custom_background(w, h)
                else:
                    bg_np = np.zeros((h, w, 3), dtype=np.uint8) + 30
                self.pipeline_gpu_bg_cache = torch.from_numpy(np.ascontiguousarray(bg_np)).to(device=device, dtype=work_dtype)
                self.pipeline_last_bg_mode = current_mode
                self.force_bg_update = False

            alpha_3 = alpha_t.unsqueeze(-1)
            output = (rgb_t * alpha_3 + self.pipeline_gpu_bg_cache * (1.0 - alpha_3)).clamp(0, 255).to(torch.uint8)

        alpha_np = alpha_u8.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()
        return output_np, alpha_np

    def _process_post_rgb_frame(self, rgb_frame, segmenter, bg_cache=None, last_bg_mode=None):
        if self._is_yolo_primary_model():
            self._run_yolo_primary(rgb_frame)
            alpha_2d = self._make_yolo_alpha(rgb_frame.shape)
        else:
            mask_binary = segmenter.predict_mask(rgb_frame)
            alpha_2d = self._postprocess_to_alpha(rgb_frame, mask_binary)
            if self.yolo_sync_postprocess.get():
                self._run_yolo_postprocess_sync(rgb_frame)
            else:
                self._schedule_yolo_postprocess(rgb_frame)
            alpha_2d = self._apply_yolo_roi_to_alpha(alpha_2d)
        compose_rgb_frame, alpha_2d = self._apply_corridor_key(rgb_frame, alpha_2d)
        output_frame, bg_cache, last_bg_mode = self._compose_processed_frame(
            compose_rgb_frame,
            alpha_2d,
            bg_cache,
            last_bg_mode
        )
        return output_frame, alpha_2d, bg_cache, last_bg_mode

    def _process_post_image(self, input_path, output_path):
        image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError("Bild konnte nicht gelesen werden.")

        rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self.model_lock:
            segmenter = self.segmenter
        output_frame, alpha_2d, _, _ = self._process_post_rgb_frame(rgb_frame, segmenter)
        if output_frame.shape[2] == 4:
            output_to_write = cv2.cvtColor(output_frame, cv2.COLOR_RGBA2BGRA)
        else:
            output_to_write = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        if not cv2.imwrite(output_path, output_to_write):
            raise RuntimeError("Ausgabebild konnte nicht geschrieben werden.")

        preview_frame = cv2.resize(rgb_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
        preview_alpha = cv2.resize(alpha_2d, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
        preview_output = cv2.resize(output_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
        self._set_latest_display(preview_frame, preview_alpha, preview_output)
        self.root.after(0, lambda: self._set_post_progress(1.0, "Bild fertig verarbeitet."))

    def _find_ffmpeg_executable(self):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        app_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(app_dir, "ffmpeg.exe"),
            os.path.join(app_dir, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(app_dir, "bin", "ffmpeg.exe"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def _open_prores_4444_writer(self, output_path, width, height, fps):
        ffmpeg_path = self._find_ffmpeg_executable()
        if not ffmpeg_path:
            raise RuntimeError(
                "FFmpeg wurde nicht gefunden. Für transparente MOV-Dateien bitte FFmpeg installieren "
                "oder ffmpeg.exe in den Projektordner bzw. in einen ffmpeg\\bin-Unterordner legen."
            )

        command = [
            ffmpeg_path,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s:v", f"{int(width)}x{int(height)}",
            "-r", f"{float(fps):.6f}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "prores_ks",
            "-profile:v", "4",
            "-pix_fmt", "yuva444p10le",
            "-alpha_bits", "16",
            "-vendor", "apl0",
            output_path,
        ]
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
        except Exception as exc:
            raise RuntimeError(f"FFmpeg konnte nicht gestartet werden: {exc}") from exc

    def _process_post_transparent_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Video konnte nicht geöffnet werden.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video-Aufloesung konnte nicht gelesen werden.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        writer = self._open_prores_4444_writer(output_path, width, height, fps)

        with self.model_lock:
            segmenter = self.segmenter

        bg_cache = None
        last_bg_mode = None
        processed = 0
        start = time.perf_counter()

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                output_frame, alpha_2d, bg_cache, last_bg_mode = self._process_post_rgb_frame(
                    rgb_frame, segmenter, bg_cache, last_bg_mode
                )
                if output_frame.shape[2] != 4:
                    alpha_u8 = self._alpha_to_u8(alpha_2d)
                    output_frame = np.dstack((rgb_frame, alpha_u8))

                try:
                    writer.stdin.write(np.ascontiguousarray(output_frame).tobytes())
                except Exception as exc:
                    raise RuntimeError(f"FFmpeg konnte keinen Frame schreiben: {exc}") from exc

                processed += 1

                if processed == 1 or processed % 10 == 0:
                    elapsed = max(0.001, time.perf_counter() - start)
                    proc_fps = processed / elapsed
                    progress = (processed / total_frames) if total_frames > 0 else 0.0
                    status = f"Exportiere ProRes 4444 Frame {processed}"
                    if total_frames > 0:
                        status += f"/{total_frames}"
                    status += f" ({proc_fps:.1f} FPS)"
                    preview_frame = cv2.resize(rgb_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    preview_alpha = cv2.resize(alpha_2d, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
                    preview_output = cv2.resize(output_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    self._set_latest_display(preview_frame, preview_alpha, preview_output)
                    self.root.after(0, lambda p=progress, s=status: self._set_post_progress(p, s))
        finally:
            cap.release()
            if writer.stdin:
                try:
                    writer.stdin.close()
                except Exception:
                    pass

        stderr = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg ProRes-Export fehlgeschlagen: {stderr.strip() or return_code}")

        self.root.after(0, lambda: self._set_post_progress(1.0, f"ProRes 4444 MOV fertig: {processed} Frames."))

    def _process_post_video(self, input_path, output_path):
        if self.bg_mode.get() == "Transparent":
            self._process_post_transparent_video(input_path, output_path)
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Video konnte nicht geöffnet werden.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video-Aufloesung konnte nicht gelesen werden.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Ausgabevideo konnte nicht erstellt werden. Nutze als Ziel am besten .mp4.")

        with self.model_lock:
            segmenter = self.segmenter

        bg_cache = None
        last_bg_mode = None
        processed = 0
        start = time.perf_counter()

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                output_frame, alpha_2d, bg_cache, last_bg_mode = self._process_post_rgb_frame(
                    rgb_frame, segmenter, bg_cache, last_bg_mode
                )
                writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                processed += 1

                if processed == 1 or processed % 10 == 0:
                    elapsed = max(0.001, time.perf_counter() - start)
                    proc_fps = processed / elapsed
                    progress = (processed / total_frames) if total_frames > 0 else 0.0
                    status = f"Verarbeite Frame {processed}"
                    if total_frames > 0:
                        status += f"/{total_frames}"
                    status += f" ({proc_fps:.1f} FPS)"
                    preview_frame = cv2.resize(rgb_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    preview_alpha = cv2.resize(alpha_2d, (self.ui_w, self.ui_h), interpolation=cv2.INTER_LINEAR)
                    preview_output = cv2.resize(output_frame, (self.ui_w, self.ui_h), interpolation=cv2.INTER_AREA)
                    self._set_latest_display(preview_frame, preview_alpha, preview_output)
                    self.root.after(0, lambda p=progress, s=status: self._set_post_progress(p, s))
        finally:
            cap.release()
            writer.release()

        self.root.after(0, lambda: self._set_post_progress(1.0, f"Video fertig: {processed} Frames."))
