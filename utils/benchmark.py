"""Compare predictions against feedbacks.pdf ground truth.

Candidates come from `resolve_ground_truth` (batch ID, then file name date). If
several remain, the result is flagged `ambiguous` and the closest entry is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from utils.categories import LABELS, label_rank
from utils.evaluator import Evaluation
from utils.feedback_parser import GroundTruth

# Recurring reviewer themes, used to check whether the AI raised the same points.
THEMES: dict[str, tuple[str, ...]] = {
    "8x8 background": ("background",),
    "homework": ("homework", "hw", "assignment"),
    "rapport / introduction": ("rapport", "intro", "greet", "welcome"),
    "individual questioning": ("individual", "one-to-one", "by name", "everyone", "each student"),
    "energy & pleasant tone": ("energetic", "energy", "pleasant", "happy", "enthusias", "tone"),
    "camera / face / lighting": ("camera", "face", "light", "visible"),
    "conclusion": ("conclusion", "recap", "wrap", "closing"),
    "new joiner introductions": ("new joiner", "new joine", "self-introduction"),
    "AV quality": ("quality issue", "audio", "clarity", "lag", "freez"),
}


@dataclass
class Comparison:
    video_id: str
    predicted: str
    ground_truth: GroundTruth
    candidates: list[GroundTruth]
    category_match: bool
    distance: int
    themes_in_gt: list[str] = field(default_factory=list)
    themes_in_ai: list[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def theme_overlap(self) -> list[str]:
        return [t for t in self.themes_in_gt if t in self.themes_in_ai]


def _themes(text: str) -> list[str]:
    text = text.lower()
    return [name for name, keywords in THEMES.items() if any(k in text for k in keywords)]


def evaluation_text(evaluation: Evaluation) -> str:
    parts = [evaluation.summary, evaluation.quality_issue_details]
    parts += evaluation.key_strengths + evaluation.areas_for_improvement
    parts += [o.observation for o in evaluation.timestamped_observations]
    return " ".join(parts)


def compare(
    video_id: str, predicted: str, candidates: list[GroundTruth], ai_text: str = ""
) -> Comparison | None:
    if not candidates:
        return None
    closest = min(
        candidates,
        key=lambda gt: abs(label_rank(gt.ground_truth_category) - label_rank(predicted)),
    )
    distance = abs(label_rank(closest.ground_truth_category) - label_rank(predicted))
    return Comparison(
        video_id=video_id,
        predicted=predicted,
        ground_truth=closest,
        candidates=candidates,
        category_match=distance == 0,
        distance=distance,
        themes_in_gt=_themes(closest.ground_truth_feedback),
        themes_in_ai=_themes(ai_text),
    )


def summarize(comparisons: list[Comparison]) -> dict:
    if not comparisons:
        return {"n": 0}
    n = len(comparisons)
    gt_theme_total = sum(len(c.themes_in_gt) for c in comparisons)
    return {
        "n": n,
        "accuracy": sum(c.category_match for c in comparisons) / n,
        "within_one": sum(c.distance <= 1 for c in comparisons) / n,
        "ambiguous": sum(c.ambiguous for c in comparisons),
        "theme_recall": (
            sum(len(c.theme_overlap) for c in comparisons) / gt_theme_total
            if gt_theme_total
            else None
        ),
    }


def confusion_matrix(comparisons: list[Comparison]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ground_truth": [c.ground_truth.ground_truth_category for c in comparisons],
            "predicted": [c.predicted for c in comparisons],
        }
    )
    matrix = pd.crosstab(df["ground_truth"], df["predicted"])
    return matrix.reindex(index=list(LABELS), columns=list(LABELS), fill_value=0)


def comparisons_table(comparisons: list[Comparison]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": c.video_id,
                "predicted": c.predicted,
                "ground_truth": c.ground_truth.ground_truth_category,
                "source_category": c.ground_truth.source_category,
                "match": c.category_match,
                "ambiguous_id": c.ambiguous,
                "shared_themes": ", ".join(c.theme_overlap),
                "missed_themes": ", ".join(t for t in c.themes_in_gt if t not in c.theme_overlap),
            }
            for c in comparisons
        ]
    )
