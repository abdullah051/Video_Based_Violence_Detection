from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from app import extract_video_frames


def make_test_video(path: Path, frame_count: int = 24) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (160, 120)
    )
    if not writer.isOpened():
        pytest.skip("MP4V video writer is unavailable in this environment")
    for index in range(frame_count):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 1] = index * 8
        writer.write(frame)
    writer.release()


def test_extract_video_frames_returns_expected_shape(tmp_path):
    video_path = tmp_path / "sample.mp4"
    make_test_video(video_path)
    frames = extract_video_frames(video_path)
    assert frames.shape == (16, 128, 128, 3)
    assert frames.dtype == np.float32
    assert np.min(frames) >= -1.0 - 1e-5
    assert np.max(frames) <= 1.0 + 1e-5


def test_extract_video_frames_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        extract_video_frames(tmp_path / "missing.mp4")


def test_extract_video_frames_rejects_too_short_video(tmp_path):
    video_path = tmp_path / "short.mp4"
    make_test_video(video_path, frame_count=8)
    with pytest.raises(ValueError, match="at least 16"):
        extract_video_frames(video_path)


def test_predict_video_creates_annotated_video(tmp_path, monkeypatch):
    from app import predict_video
    import app as app_module

    video_path = tmp_path / "to_annotate.mp4"
    make_test_video(video_path)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path / "outputs")

    class DummyModel:
        def predict(self, batch, verbose=0):
            assert batch.shape == (1, 16, 128, 128, 3)
            return np.asarray([[0.05, 0.95]], dtype=np.float32)

    output_path, summary = predict_video(str(video_path), DummyModel())
    assert output_path is not None
    assert Path(output_path).is_file()
    assert "Violence" in summary
    assert "audio track is not preserved" in summary
