"""Teaching criteria taken from the reviewer feedback, and their labels parsed from feedback text."""

from __future__ import annotations

import re
from dataclasses import dataclass

NEGATIVE_RE = re.compile(r"\b(need|needs|should|without|can't|cannot|can|no|not|turned off)\b", re.I)
CLAUSE_SPLIT_RE = re.compile(r",|\bbut\b", re.I)


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    pattern: str
    features: tuple[str, ...]
    # Reviewers mostly mention these only when lacking, so silence counts as met.
    unmentioned_met: bool = False
    exclude: str | None = None


TEACHING_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "rapport_intro",
        "Builds rapport with a proper introduction",
        r"rapport|\bintro\b|\bintroduction\b",
        ("rapport_cues_opening", "tutor_talk_ratio"),
        exclude=r"new join|self-introduction",
    ),
    Criterion("homework", "Assigns homework", r"\bhws?\b|home ?work", ("homework_mentioned",)),
    Criterion(
        "energetic_tone",
        "Energetic, pleasant interaction",
        r"energetic|pleasant|happ",
        ("tutor_loudness_std", "student_turns_per_min"),
        unmentioned_met=True,
    ),
    Criterion(
        "tutor_camera",
        "Tutor camera on and face visible",
        r"camera|face",
        ("tutor_face_ratio", "low_detail_frame_ratio"),
        unmentioned_met=True,
    ),
    Criterion(
        "new_joiner_intro",
        "Gives new joiners time to introduce themselves",
        r"new join|self-introduction",
        ("new_joiner_intro",),
        unmentioned_met=True,
    ),
)

QUALITY_ISSUE_FEATURES: tuple[str, ...] = (
    "dark_frame_ratio",
    "low_detail_frame_ratio",
    "sharpness_log",
    "tutor_face_ratio",
    "audio_silence_ratio",
)


def criterion_labels(feedback: str) -> dict[str, int | None]:
    """1 = met, 0 = criticised, None = unknown. Any criticism outweighs praise."""
    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(feedback) if c.strip()]
    labels: dict[str, int | None] = {}
    for criterion in TEACHING_CRITERIA:
        votes = [
            0 if NEGATIVE_RE.search(clause) else 1
            for clause in clauses
            if re.search(criterion.pattern, clause, re.I)
            and not (criterion.exclude and re.search(criterion.exclude, clause, re.I))
        ]
        if votes:
            labels[criterion.name] = min(votes)
        else:
            labels[criterion.name] = 1 if criterion.unmentioned_met else None
    return labels
