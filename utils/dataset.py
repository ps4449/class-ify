"""Discover sample recordings under data/<category>/ and pair each with its transcript."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from utils.categories import SOURCE_TO_LABEL
from utils.feedback_parser import extract_video_id

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
TRANSCRIPT_EXTS = {".vtt", ".txt"}
NOISE_TOKENS = {"part", "transcript", "1", "copy"}


@dataclass
class Sample:
    folder: str
    video_path: Path
    transcript_path: Path | None
    video_id: str | None

    @property
    def label(self) -> str:
        return f"{self.folder} / {self.video_path.stem.strip()}"

    @property
    def key(self) -> str:
        return f"{self.folder}/{self.video_path.name}"

    @property
    def quality_label(self) -> str | None:
        return SOURCE_TO_LABEL.get(self.folder)


def _tokens(name: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    return {t for t in tokens if t not in NOISE_TOKENS and not re.fullmatch(r"\d{2,4}", t)}


def _match_transcript(video: Path, transcripts: list[Path]) -> Path | None:
    # File names differ between video and transcript, so pick the best token overlap.
    scored = [(len(_tokens(video.stem) & _tokens(t.stem)), t) for t in transcripts]
    scored = [s for s in scored if s[0] > 0]
    return max(scored, key=lambda s: s[0])[1] if scored else None


def discover_samples(data_dir: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return samples
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = list(folder.iterdir())
        videos = sorted(f for f in files if f.suffix.lower() in VIDEO_EXTS)
        transcripts = [f for f in files if f.suffix.lower() in TRANSCRIPT_EXTS]
        for video in videos:
            samples.append(
                Sample(
                    folder=folder.name.strip(),
                    video_path=video,
                    transcript_path=_match_transcript(video, transcripts),
                    video_id=extract_video_id(video.name),
                )
            )
    return samples
