"""Local speech-to-text with faster-whisper, used when no transcript is supplied."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from utils.transcript_processor import Cue, Transcript

DEFAULT_WHISPER_MODEL = "small"


@lru_cache(maxsize=2)
def _load_model(size: str) -> WhisperModel:
    return WhisperModel(size, device="auto", compute_type="int8")


def transcribe(video_path: str | Path, model_size: str = DEFAULT_WHISPER_MODEL) -> Transcript:
    segments, _ = _load_model(model_size).transcribe(str(video_path), vad_filter=True)
    # Whisper has no speaker labels, so speaker-based features will be missing.
    return Transcript([Cue(s.start, s.end, None, s.text.strip()) for s in segments])
