"""Local video-level violence classification demo using a trained Keras model.

The model predicts one label for the whole clip from 16 uniformly sampled frames.
It is not a real-time, frame-by-frame violence detector.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "violence_model.h5"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "outputs"))
CLASS_NAMES = ("NonViolence", "Violence")
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


def extract_video_frames(video_path: str | os.PathLike[str]) -> np.ndarray:
    """Sample 16 frames uniformly and apply MobileNetV2 preprocessing.

    Returns an array shaped (16, 128, 128, 3), float32, in [-1, 1].
    This matches the preprocessing used in the training notebook.
    """
    path = Path(video_path)
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open this video. Try another MP4 or AVI file.")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < SEQUENCE_LENGTH:
            raise ValueError(
                f"The video must contain at least {SEQUENCE_LENGTH} readable frames. "
                f"This file reports {total_frames}."
            )

        indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=np.int64)
        frames = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read video frame {int(frame_index)}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            # Equivalent to keras.applications.mobilenet_v2.preprocess_input for RGB uint8 data.
            frame = frame.astype(np.float32) / 127.5 - 1.0
            frames.append(frame)
    finally:
        capture.release()

    if len(frames) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, but extracted {len(frames)}.")
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def load_trained_model(model_path: Path = MODEL_PATH):
    """Load the provided HDF5 model without restoring its training optimizer."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Place violence_model.h5 in the models/ folder."
        )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install the dependencies with: pip install -r requirements.txt"
        ) from exc

    return tf.keras.models.load_model(model_path, compile=False)


def annotate_video(
    video_path: str | os.PathLike[str],
    label: str,
    confidence: float,
) -> str:
    """Write a silent annotated copy, overlaying the clip-level prediction on every frame."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Could not reopen the uploaded video for annotation.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video has invalid dimensions.")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"annotated_{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("Could not create the output MP4. Check OpenCV video codec support.")

    color = (0, 0, 255) if label == "Violence" else (0, 200, 0)
    overlay = f"{label} | {confidence:.1%} (video-level)"
    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.rectangle(frame, (12, 12), (min(width - 12, 610), 66), (20, 20, 20), -1)
            cv2.putText(
                frame, overlay, (24, 49), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, color, 2, cv2.LINE_AA
            )
            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        writer.release()

    if frames_written == 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were written to the annotated video.")
    return str(output_path)


def predict_video(video_path: str | None, model) -> Tuple[str | None, str]:
    """Predict one label for the clip and return the annotated video and summary."""
    if not video_path:
        return None, "Please upload a video first."

    frames = extract_video_frames(video_path)
    batch = np.expand_dims(frames, axis=0)
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    if probabilities.shape != (len(CLASS_NAMES),) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("The model returned an unexpected prediction. Check the model file.")

    class_index = int(np.argmax(probabilities))
    label = CLASS_NAMES[class_index]
    confidence = float(probabilities[class_index])
    output_path = annotate_video(video_path, label, confidence)
    summary = (
        f"Prediction: {label}\n"
        f"Confidence: {confidence:.2%}\n\n"
        f"Method: one video-level prediction based on {SEQUENCE_LENGTH} uniformly sampled frames.\n"
        "Note: the output is a silent annotated copy; the original audio track is not preserved."
    )
    return output_path, summary


def main() -> None:
    model = load_trained_model()
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Run: pip install -r requirements.txt") from exc

    demo = gr.Interface(
        fn=lambda video: predict_video(video, model),
        inputs=gr.Video(sources=["upload"], label="Upload a video"),
        outputs=[
            gr.Video(label="Annotated video", format="mp4"),
            gr.Textbox(label="Prediction summary", lines=5),
        ],
        title="Video Violence Classification",
        description=(
            "Upload a video to classify the clip as Violence or NonViolence. "
            "This is a research prototype, not a real-time detector or a decision-making system."
        ),
    )
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() == "true"
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    if share:
        print("WARNING: GRADIO_SHARE=true creates a publicly accessible temporary share link.")
    demo.launch(server_name=server_name, server_port=server_port, share=share, inbrowser=True)


if __name__ == "__main__":
    main()
