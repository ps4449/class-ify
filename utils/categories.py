LABELS: tuple[str, ...] = ("poor", "average", "good", "excellent")

SOURCE_TO_LABEL: dict[str, str] = {
    "weak": "poor",
    "quality-issue": "poor",
    "average": "average",
    "good": "good",
    "excellent": "excellent",
}

LABEL_COLORS: dict[str, str] = {
    "poor": "#d64545",
    "average": "#e0a526",
    "good": "#3b8fd9",
    "excellent": "#2e9e5b",
}


def label_rank(label: str) -> int:
    return LABELS.index(label)
