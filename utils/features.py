"""Local transcript, visual and audio features. Missing values are NaN and imputed by the classifier."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from utils.redaction import STUDENT_PREFIX, TUTOR
from utils.transcript_processor import Transcript
from utils.video_processor import Keyframe

NAN = float("nan")

RAPPORT_RE = re.compile(r"how are you|good (morning|afternoon|evening)|welcome|how was your|what did you do", re.I)
INTRO_RE = re.compile(r"introduce (yourself|yourselves)|tell (me|us) about yourself|your name", re.I)
HOMEWORK_RE = re.compile(r"home ?work|\bhw\b|assignment|practi[cs]e at home", re.I)
CONCLUSION_RE = re.compile(r"recap|summar|what did we learn|that'?s all for today|see you (next|on)|next class", re.I)
STUDENT_REF_RE = re.compile(rf"{STUDENT_PREFIX} \d+")

LONG_GAP_S = 60
TUTOR_SPEECH_MARGIN_S = 2
DARK_THRESHOLD = 50
LOW_DETAIL_THRESHOLD = 15
SILENCE_DB = -45.0
SAMPLE_RATE = 16000

TRANSCRIPT_FEATURES: dict[str, str] = {
    "tutor_talk_ratio": "Share of speaking time taken by the tutor (0-1)",
    "students_speaking": "Number of distinct students who spoke",
    "student_turn_balance": "Least / most active student's turn count (1 = evenly included)",
    "students_addressed_ratio": "Share of speaking students the tutor addressed by name",
    "tutor_questions_per_min": "Questions asked by the tutor per minute",
    "student_turns_per_min": "Student speaking turns per minute",
    "rapport_cues_opening": "Greeting / rapport phrases in the first 5 minutes",
    "new_joiner_intro": "Students invited to introduce themselves (0/1)",
    "homework_mentioned": "Homework mentioned during the class (0/1)",
    "conclusion_cues": "Recap / closing phrases in the last 10% of the class",
    "long_gaps_per_hour": f"Gaps longer than {LONG_GAP_S}s without speech, per hour",
}
VISUAL_FEATURES: dict[str, str] = {
    "tutor_face_ratio": "Share of frames during tutor speech with a visible face (tutor camera on)",
    "brightness_mean": "Mean frame brightness (0-255)",
    "dark_frame_ratio": "Share of dark frames",
    "low_detail_frame_ratio": "Share of near-blank frames (camera off / frozen)",
    "sharpness_log": "Log Laplacian variance (higher = sharper video)",
}
AUDIO_FEATURES: dict[str, str] = {
    "audio_silence_ratio": "Share of 1-second windows that are silent",
    "tutor_loudness_db": "Median loudness while the tutor speaks (dBFS)",
    "tutor_loudness_std": "Loudness variation while the tutor speaks (vocal energy)",
}
FEATURE_DESCRIPTIONS = {**TRANSCRIPT_FEATURES, **VISUAL_FEATURES, **AUDIO_FEATURES}
FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)


def _empty(group: dict[str, str]) -> dict[str, float]:
    return {name: NAN for name in group}


def _tutor_intervals(transcript: Transcript | None) -> list[tuple[float, float]]:
    """Tutor speaking intervals; falls back to all cues when there are no speaker labels."""
    if not transcript or not transcript.cues:
        return []
    tutor = [c for c in transcript.cues if c.speaker == TUTOR] or transcript.cues
    return [(c.start, c.end) for c in tutor]


def transcript_features(transcript: Transcript | None, duration: float) -> dict[str, float]:
    f = _empty(TRANSCRIPT_FEATURES)
    if not transcript or not transcript.cues:
        return f
    cues = transcript.cues
    minutes = max(duration, 60) / 60
    tutor = [c for c in cues if c.speaker == TUTOR]
    students = [c for c in cues if c.speaker and c.speaker.startswith(STUDENT_PREFIX)]
    tutor_text = " ".join(c.text for c in (tutor or cues))

    if tutor:
        tutor_time = sum(c.end - c.start for c in tutor)
        student_time = sum(c.end - c.start for c in students)
        turns = Counter(c.speaker for c in students)
        f["tutor_talk_ratio"] = tutor_time / max(tutor_time + student_time, 1e-6)
        f["students_speaking"] = len(turns)
        f["student_turn_balance"] = min(turns.values()) / max(turns.values()) if len(turns) > 1 else NAN
        addressed = set(STUDENT_REF_RE.findall(tutor_text)) & set(turns)
        f["students_addressed_ratio"] = len(addressed) / len(turns) if turns else NAN
        f["student_turns_per_min"] = len(students) / minutes

    f["tutor_questions_per_min"] = tutor_text.count("?") / minutes

    start, end = cues[0].start, max(duration, cues[-1].end)
    opening = [c for c in cues if c.start < start + 300]
    last_tenth = [c for c in cues if c.start > end - 0.10 * (end - start)]
    f["rapport_cues_opening"] = sum(bool(RAPPORT_RE.search(c.text)) for c in opening)
    f["new_joiner_intro"] = float(any(INTRO_RE.search(c.text) for c in cues))
    f["homework_mentioned"] = float(any(HOMEWORK_RE.search(c.text) for c in cues))
    f["conclusion_cues"] = sum(bool(CONCLUSION_RE.search(c.text)) for c in last_tenth)

    gaps = sum(1 for a, b in zip(cues, cues[1:]) if b.start - a.end > LONG_GAP_S)
    f["long_gaps_per_hour"] = gaps / (minutes / 60)
    return f


_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def visual_features(keyframes: list[Keyframe], transcript: Transcript | None = None) -> dict[str, float]:
    if not keyframes:
        return _empty(VISUAL_FEATURES)
    intervals = _tutor_intervals(transcript)
    brightness, contrast, sharpness, tutor_face = [], [], [], []
    for frame in keyframes:
        gray = cv2.cvtColor(np.array(frame.image), cv2.COLOR_RGB2GRAY)
        brightness.append(gray.mean())
        contrast.append(gray.std())
        sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        # The active-speaker tile shows the tutor while they talk, so only those frames count.
        t = frame.timestamp
        if not intervals or any(s - TUTOR_SPEECH_MARGIN_S <= t <= e + TUTOR_SPEECH_MARGIN_S for s, e in intervals):
            faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            tutor_face.append(len(faces) > 0)
    return {
        "tutor_face_ratio": float(np.mean(tutor_face)) if tutor_face else NAN,
        "brightness_mean": float(np.mean(brightness)),
        "dark_frame_ratio": float(np.mean(np.array(brightness) < DARK_THRESHOLD)),
        "low_detail_frame_ratio": float(np.mean(np.array(contrast) < LOW_DETAIL_THRESHOLD)),
        "sharpness_log": float(math.log1p(np.median(sharpness))),
    }


def audio_features(video_path: str | Path, transcript: Transcript | None = None) -> dict[str, float]:
    import av
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(str(video_path), sampling_rate=SAMPLE_RATE)
    except (av.error.FFmpegError, IndexError):  # unreadable media or no audio stream
        return _empty(AUDIO_FEATURES)
    windows = audio[: len(audio) // SAMPLE_RATE * SAMPLE_RATE].reshape(-1, SAMPLE_RATE)
    if not len(windows):
        return _empty(AUDIO_FEATURES)
    db = 20 * np.log10(np.sqrt((windows**2).mean(axis=1)) + 1e-9)

    tutor_mask = np.zeros(len(db), dtype=bool)
    for start, end in _tutor_intervals(transcript):
        tutor_mask[int(start) : int(math.ceil(end))] = True
    if not tutor_mask.any():
        tutor_mask[:] = True
    tutor_db = db[tutor_mask & (db > SILENCE_DB)]
    return {
        "audio_silence_ratio": float((db <= SILENCE_DB).mean()),
        "tutor_loudness_db": float(np.median(tutor_db)) if len(tutor_db) else NAN,
        "tutor_loudness_std": float(tutor_db.std()) if len(tutor_db) else NAN,
    }


def format_features(features: dict[str, float]) -> str:
    lines = []
    for name in FEATURE_NAMES:
        value = features.get(name, NAN)
        shown = "unavailable" if value is None or math.isnan(value) else f"{value:.3g}"
        lines.append(f"- {name} = {shown}  ({FEATURE_DESCRIPTIONS[name]})")
    return "\n".join(lines)
