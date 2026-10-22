"""Criterion-level classifiers trained offline by train.py.

Each teaching criterion gets a small model on a few signals, and a fixed rubric turns
criterion probabilities into a teaching rating. AV quality issues are predicted separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.categories import label_rank
from utils.criteria import QUALITY_ISSUE_FEATURES, TEACHING_CRITERIA
from utils.features import FEATURE_NAMES

MODEL_PATH = Path("models/classifier.joblib")
QUALITY_ISSUE = "quality_issue"
TEACHING = "teaching"
# Hand-set cut-offs on the mean criterion probability; not fitted to the data.
RATING_THRESHOLDS = ((0.9, "excellent"), (0.7, "good"), (0.45, "average"))


@dataclass
class ClassifierPrediction:
    teaching_label: str
    teaching_score: float
    criteria: dict[str, float]  # P(criterion met)
    quality_issue: float  # P(AV quality issue)


@dataclass
class BinaryModel:
    features: tuple[str, ...]
    pipeline: Pipeline | None
    constant: float  # used when training labels contain a single class

    def predict(self, row: pd.DataFrame) -> float:
        if self.pipeline is None:
            return self.constant
        return float(self.pipeline.predict_proba(row[list(self.features)])[0, 1])


def _fit_binary(X: pd.DataFrame, y: np.ndarray, features: tuple[str, ...]) -> BinaryModel:
    if len(set(y)) < 2:
        return BinaryModel(features, None, float(y.mean()) if len(y) else 0.5)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
        ]
    )
    return BinaryModel(features, pipeline.fit(X[list(features)], y), 0.0)


def rating_from_score(score: float) -> str:
    return next((label for cutoff, label in RATING_THRESHOLDS if score >= cutoff), "poor")


def _to_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=FEATURE_NAMES).astype(float)


def _fit_models(
    X: pd.DataFrame, criteria: list[dict[str, int | None]], quality: list[int], mask: np.ndarray
) -> dict[str, BinaryModel]:
    models = {}
    for c in TEACHING_CRITERIA:
        y = np.array([-1 if row[c.name] is None else row[c.name] for row in criteria])
        known = mask & (y >= 0)
        models[c.name] = _fit_binary(X[known], y[known], c.features)
    models[QUALITY_ISSUE] = _fit_binary(X[mask], np.array(quality)[mask], QUALITY_ISSUE_FEATURES)
    return models


def _predict(models: dict[str, BinaryModel], row: pd.DataFrame) -> ClassifierPrediction:
    criteria = {c.name: models[c.name].predict(row) for c in TEACHING_CRITERIA}
    score = float(np.mean(list(criteria.values())))
    return ClassifierPrediction(rating_from_score(score), score, criteria, models[QUALITY_ISSUE].predict(row))


def _score(pairs: list[tuple]) -> dict:
    if not pairs:
        return {"n": 0, "accuracy": None, "baseline": None}
    truth = [t for t, _ in pairs]
    return {
        "n": len(pairs),
        "accuracy": float(np.mean([t == p for t, p in pairs])),
        "baseline": max(Counter(truth).values()) / len(truth),
    }


def train(
    keys: list[str],
    rows: list[dict[str, float]],
    criteria: list[dict[str, int | None]],
    teaching: list[str | None],
    quality: list[int],
) -> dict:
    """`teaching` is None for quality-issue recordings, which are excluded from the rating metric."""
    X = _to_frame(rows)
    n = len(keys)

    # Leave-one-out: each sample is scored by models that never saw it. These
    # predictions are also reused at inference for training samples to avoid leakage.
    loocv = {}
    for i in range(n):
        loocv[keys[i]] = _predict(_fit_models(X, criteria, quality, np.arange(n) != i), X.iloc[[i]])

    metrics = {
        c.name: _score(
            [(row[c.name], int(loocv[k].criteria[c.name] >= 0.5)) for k, row in zip(keys, criteria) if row[c.name] is not None]
        )
        for c in TEACHING_CRITERIA
    }
    rating_pairs = [(t, loocv[k].teaching_label) for k, t in zip(keys, teaching) if t]
    metrics[TEACHING] = _score(rating_pairs)
    metrics[TEACHING]["within_one"] = (
        float(np.mean([abs(label_rank(t) - label_rank(p)) <= 1 for t, p in rating_pairs])) if rating_pairs else None
    )
    metrics[QUALITY_ISSUE] = _score([(q, int(loocv[k].quality_issue >= 0.5)) for k, q in zip(keys, quality)])

    return {
        "models": _fit_models(X, criteria, quality, np.ones(n, dtype=bool)),
        "loocv": loocv,
        "metrics": metrics,
        "n_samples": n,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def is_reliable(metric: dict) -> bool:
    return metric["accuracy"] is not None and metric["accuracy"] > metric["baseline"]


def reliable_parts(bundle: dict | None) -> set[str]:
    """Parts whose LOOCV accuracy beats always guessing the majority label."""
    return {name for name, metric in bundle["metrics"].items() if is_reliable(metric)} if bundle else set()


def prompt_estimates(prediction: ClassifierPrediction, bundle: dict | None) -> list[str]:
    parts = reliable_parts(bundle)
    lines = []
    if TEACHING in parts:
        lines.append(f"Teaching rating estimate: {prediction.teaching_label} (rubric score {prediction.teaching_score:.2f})")
    for c in TEACHING_CRITERIA:
        if c.name in parts:
            lines.append(f"P({c.description}) = {prediction.criteria[c.name]:.2f}")
    if QUALITY_ISSUE in parts:
        lines.append(f"P(audio/video quality issue) = {prediction.quality_issue:.2f}")
    return lines


def save_bundle(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return bundle if "metrics" in bundle else None  # model from an older format; retrain


def predict(bundle: dict, features: dict[str, float]) -> ClassifierPrediction:
    return _predict(bundle["models"], _to_frame([features]))
