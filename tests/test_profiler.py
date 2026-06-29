from ai_segmenter.profiler import PipelineProfiler


def test_pipeline_profiler_writes_summary(tmp_path):
    profiler = PipelineProfiler(log_dir=str(tmp_path), interval_s=0.0)
    path = profiler.start(
        {
            "source": "test",
            "model": "MediaPipe Selfie",
            "backend": "unit",
            "capture_width": 640,
            "capture_height": 360,
            "source_fps": 25.0,
        }
    )

    profiler.count("capture")
    profiler.sample("ai", 0.01)
    profiler.flush_if_due(force=True)
    summary = profiler.get_last_summary()
    profiler.close()

    assert summary is not None
    assert summary["source"] == "test"
    assert summary["model"] == "MediaPipe Selfie"
    assert "capture_fps" in summary
    assert path.endswith(".csv")

