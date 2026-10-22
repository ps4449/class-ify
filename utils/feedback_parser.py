"""Parse ground-truth feedback from feedbacks.pdf, keyed by batch ID (e.g. "B435").

A batch ID can appear more than once, so each key maps to a list of entries;
`resolve_ground_truth` disambiguates those by the date in the video file name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from pypdf import PdfReader

from utils.categories import SOURCE_TO_LABEL

SECTION_RE = re.compile(
    r"\*\s*(excellent|good|average|weak|known quality issues)\s*-?\s*\*", re.IGNORECASE
)
ENTRY_RE = re.compile(
    r"\d+\.\s*(B\d+)\s*\(([^)]*)\)\s*(.*?)\s*-{2,}\s*(.*?)(?=\s*\d+\.\s*B\d+\s*\(|\Z)",
    re.DOTALL,
)
VIDEO_ID_RE = re.compile(r"^\s*(B\d+)", re.IGNORECASE)
VIDEO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
PDF_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y")

SECTION_TO_SOURCE = {
    "excellent": "excellent",
    "good": "good",
    "average": "average",
    "weak": "weak",
    "known quality issues": "quality-issue",
}


@dataclass
class GroundTruth:
    video_id: str
    source_category: str
    ground_truth_category: str
    ground_truth_feedback: str
    date: str
    parsed_date: date | None  # None when the PDF date is malformed
    tutor: str


def extract_video_id(filename: str | Path) -> str | None:
    match = VIDEO_ID_RE.match(Path(filename).name)
    return match.group(1).upper() if match else None


def extract_video_date(filename: str | Path) -> date | None:
    match = VIDEO_DATE_RE.search(Path(filename).name)
    if not match:
        return None
    try:
        return date(*map(int, match.groups()))
    except ValueError:
        return None


def _parse_pdf_date(text: str) -> date | None:
    for fmt in PDF_DATE_FORMATS:
        try:
            return datetime.strptime(text.replace(" ", ""), fmt).date()
        except ValueError:
            continue
    return None


def resolve_ground_truth(
    lookup: dict[str, list[GroundTruth]], filename: str | Path
) -> list[GroundTruth]:
    """Entries for the file's batch ID, narrowed by the file name date when the ID is repeated."""
    candidates = lookup.get(extract_video_id(filename) or "", [])
    video_date = extract_video_date(filename)
    if len(candidates) > 1 and video_date:
        dated = [c for c in candidates if c.parsed_date == video_date]
        if dated:
            return dated
    return candidates


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse_feedback_text(text: str) -> dict[str, list[GroundTruth]]:
    lookup: dict[str, list[GroundTruth]] = {}
    sections = list(SECTION_RE.finditer(text))
    for i, section in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        source = SECTION_TO_SOURCE[_clean(section.group(1)).lower()]
        for entry in ENTRY_RE.finditer(text[section.end() : end]):
            video_id, date_text, tutor, feedback = entry.groups()
            gt = GroundTruth(
                video_id=video_id.upper(),
                source_category=source,
                ground_truth_category=SOURCE_TO_LABEL[source],
                ground_truth_feedback=_clean(feedback),
                date=_clean(date_text),
                parsed_date=_parse_pdf_date(date_text),
                tutor=_clean(tutor),
            )
            lookup.setdefault(gt.video_id, []).append(gt)
    return lookup


def parse_feedback_pdf(pdf_path: str | Path) -> dict[str, list[GroundTruth]]:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_feedback_text(text)
