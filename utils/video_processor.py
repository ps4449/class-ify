"""Sample keyframes from a recording; the interval stretches so at most `max_frames` are kept."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image

# Shared by train.py and the app; changing them requires retraining the classifier.
DEFAULT_TARGET_INTERVAL_S = 20
DEFAULT_MAX_FRAMES = 60
DEFAULT_MAX_WIDTH = 768


@dataclass
class Keyframe:
    timestamp: float
    image: Image.Image


@dataclass
class VideoSample:
    duration: float
    interval: float
    keyframes: list[Keyframe]


def _resize(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    height = round(image.height * max_width / image.width)
    return image.resize((max_width, height), Image.LANCZOS)


def extract_keyframes(
    video_path: str | Path,
    target_interval_s: float = DEFAULT_TARGET_INTERVAL_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> VideoSample:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if frame_count > 0 else 0.0
        if duration <= 0:
            raise ValueError(f"Could not determine duration of: {video_path}")

        interval = max(target_interval_s, duration / max_frames)
        keyframes: list[Keyframe] = []
        t = interval / 2
        while t < duration and len(keyframes) < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.append(Keyframe(t, _resize(Image.fromarray(rgb), max_width)))
            t += interval
        return VideoSample(duration=duration, interval=interval, keyframes=keyframes)
    finally:
        cap.release()
