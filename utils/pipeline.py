"""Local analysis (frames, Whisper, redaction, features, classifier) -> Gemini on redacted text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google import genai

from utils.classifier import ClassifierPrediction, predict, prompt_estimates
from utils.evaluator import DEFAULT_MODEL, Evaluation, build_contents, evaluate_class
from utils.features import audio_features, transcript_features, visual_features
from utils.feedback_parser import extract_video_id
from utils.redaction import redact_transcript
from utils.transcriber import DEFAULT_WHISPER_MODEL, transcribe
from utils.transcript_processor import Transcript
from utils.video_processor import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_TARGET_INTERVAL_S,
    Keyframe,
    extract_keyframes,
)

CACHE_DIR = Path(".cache")
FEATURES_VERSION = 3


@dataclass
class LocalAnalysis:
    features: dict[str, float]
    transcript: Transcript | None  # redacted
    keyframes: list[Keyframe]
    duration: float
    transcribed: bool


@dataclass
class PipelineResult:
    video_id: str | None
    video_name: str
    local: LocalAnalysis
    classifier: ClassifierPrediction | None
    evaluation: Evaluation
    model_used: str
    sent_to_gemini: str


def analyze_locally(
    video_path: str | Path,
    transcript: Transcript | None = None,
    *,
    target_interval_s: float = DEFAULT_TARGET_INTERVAL_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    on_step: Callable[[str], None] = lambda _: None,
) -> LocalAnalysis:
    on_step("Extracting keyframes")
    video = extract_keyframes(video_path, target_interval_s, max_frames)

    transcribed = False
    if transcript is None or not transcript.cues:
        on_step(f"Transcribing audio locally (Whisper {whisper_model})")
        transcript = transcribe(video_path, whisper_model)
        transcribed = True

    on_step("Redacting names")
    redacted = redact_transcript(transcript)

    on_step("Computing local features")
    features = {
        **transcript_features(redacted, video.duration),
        **visual_features(video.keyframes, redacted),
        **audio_features(video_path, redacted),
    }
    return LocalAnalysis(features, redacted, video.keyframes, video.duration, transcribed)


def run_pipeline(
    client: genai.Client,
    video_path: str | Path,
    transcript: Transcript | None = None,
    *,
    model: str = DEFAULT_MODEL,
    fallback_models: list[str] | None = None,
    classifier_bundle: dict | None = None,
    classifier_override: ClassifierPrediction | None = None,
    video_name: str | None = None,
    on_step: Callable[[str], None] = lambda _: None,
    **local_kwargs,
) -> PipelineResult:
    local = analyze_locally(video_path, transcript, on_step=on_step, **local_kwargs)

    classifier = classifier_override
    if classifier is None and classifier_bundle:
        classifier = predict(classifier_bundle, local.features)

    on_step(f"Evaluating redacted transcript with {model}")
    estimates = prompt_estimates(classifier, classifier_bundle) if classifier else []
    contents = build_contents(local.transcript, local.features, estimates, local.duration)
    evaluation, model_used = evaluate_class(client, contents, model, fallback_models)

    video_name = video_name or Path(video_path).name
    return PipelineResult(
        video_id=extract_video_id(video_name),
        video_name=video_name,
        local=local,
        classifier=classifier,
        evaluation=evaluation,
        model_used=model_used,
        sent_to_gemini="\n\n".join(contents),
    )


def cache_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _cache_path(kind: str, key: str) -> Path:
    return CACHE_DIR / kind / f"{key}.json"


def load_cached_features(key: str) -> dict[str, float] | None:
    path = _cache_path("features", key)
    return json.loads(path.read_text()) if path.exists() else None


def save_cached_features(key: str, features: dict[str, float]) -> None:
    path = _cache_path("features", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, indent=2))


def load_cached(key: str) -> tuple[Evaluation, str] | None:
    path = _cache_path("evaluations", key)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Evaluation.model_validate(data["evaluation"]), data["model_used"]


def save_cached(key: str, evaluation: Evaluation, model_used: str) -> None:
    path = _cache_path("evaluations", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_used": model_used, "evaluation": evaluation.model_dump()}, indent=2))
