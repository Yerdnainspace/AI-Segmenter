import csv
import os
import threading
import time


class PipelineProfiler:
    COUNT_STAGES = (
        "capture",
        "source_arrival",
        "source_overwrite",
        "ai_input_overwrite",
        "output_overwrite",
        "preprocess",
        "ai",
        "alpha_post",
        "yolo_post",
        "corridor",
        "compose",
        "display",
        "decklink_output",
        "decklink_repeat",
        "frame_done",
        "read_fail",
    )
    TIME_STAGES = (
        "capture_read",
        "capture_actual",
        "capture_interval",
        "preprocess",
        "ai",
        "alpha_post",
        "yolo_post",
        "corridor",
        "compose",
        "display",
        "decklink_output",
        "total",
        "compute_total",
    )

    def __init__(self, log_dir="logs", interval_s=1.0):
        self.interval_s = float(interval_s)
        self.log_dir = log_dir
        self.path = None
        self.file = None
        self.writer = None
        self.lock = threading.RLock()
        self.started_at = 0.0
        self.window_started_at = 0.0
        self.metadata = {}
        self.last_summary = None
        self.counts = {stage: 0 for stage in self.COUNT_STAGES}
        self.time_sums = {stage: 0.0 for stage in self.TIME_STAGES}
        self.time_counts = {stage: 0 for stage in self.TIME_STAGES}

    def start(self, metadata):
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.abspath(os.path.join(self.log_dir, f"pipeline_{stamp}.csv"))
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        headers = [
            "timestamp",
            "elapsed_s",
            "interval_s",
            "source",
            "model",
            "model_device",
            "model_dtype",
            "model_backend",
            "backend",
            "capture_width",
            "capture_height",
            "source_fps",
        ]
        headers.extend(f"{stage}_fps" for stage in self.COUNT_STAGES)
        headers.extend(f"{stage}_avg_ms" for stage in self.TIME_STAGES)
        self.writer = csv.DictWriter(self.file, fieldnames=headers)
        self.writer.writeheader()
        now = time.perf_counter()
        self.started_at = now
        self.window_started_at = now
        self.metadata = dict(metadata or {})
        return self.path

    def count(self, stage, amount=1):
        if stage not in self.counts:
            return
        with self.lock:
            self.counts[stage] += int(amount)

    def sample(self, stage, duration_s):
        if stage not in self.time_sums:
            return
        with self.lock:
            self.time_sums[stage] += float(duration_s)
            self.time_counts[stage] += 1

    def update_metadata(self, **metadata):
        with self.lock:
            self.metadata.update(metadata)

    def flush_if_due(self, force=False):
        with self.lock:
            if self.writer is None or self.file is None:
                return
            now = time.perf_counter()
            interval = now - self.window_started_at
            if not force and interval < self.interval_s:
                return
            if force and not any(self.counts.values()) and not any(self.time_counts.values()):
                return
            if interval <= 0:
                return

            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s": f"{now - self.started_at:.3f}",
                "interval_s": f"{interval:.3f}",
                "source": self.metadata.get("source", ""),
                "model": self.metadata.get("model", ""),
                "model_device": self.metadata.get("model_device", ""),
                "model_dtype": self.metadata.get("model_dtype", ""),
                "model_backend": self.metadata.get("model_backend", ""),
                "backend": self.metadata.get("backend", ""),
                "capture_width": self.metadata.get("capture_width", 0),
                "capture_height": self.metadata.get("capture_height", 0),
                "source_fps": f"{float(self.metadata.get('source_fps', 0.0) or 0.0):.3f}",
            }
            for stage in self.COUNT_STAGES:
                row[f"{stage}_fps"] = f"{self.counts[stage] / interval:.3f}"
            for stage in self.TIME_STAGES:
                count = self.time_counts[stage]
                avg_ms = (self.time_sums[stage] / count * 1000.0) if count else 0.0
                row[f"{stage}_avg_ms"] = f"{avg_ms:.3f}"
            self.writer.writerow(row)
            self.file.flush()
            self.last_summary = dict(row)

            self.window_started_at = now
            self.counts = {stage: 0 for stage in self.COUNT_STAGES}
            self.time_sums = {stage: 0.0 for stage in self.TIME_STAGES}
            self.time_counts = {stage: 0 for stage in self.TIME_STAGES}

    def get_last_summary(self):
        with self.lock:
            return dict(self.last_summary) if self.last_summary else None

    def close(self):
        with self.lock:
            if self.writer is not None:
                self.flush_if_due(force=True)
            if self.file is not None:
                self.file.close()
            self.writer = None
            self.file = None
