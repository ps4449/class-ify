"""Load WebVTT transcripts with "Speaker: text" cues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TIMING_RE = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")
TAG_RE = re.compile(r"<[^>]+>")
MAX_SPEAKER_LEN = 60


@dataclass
class Cue:
    start: float
    end: float
    speaker: str | None
    text: str


@dataclass
class Transcript:
    cues: list[Cue]

    @property
    def duration(self) -> float:
        return self.cues[-1].end if self.cues else 0.0

    @property
    def speakers(self) -> list[str]:
        return sorted({c.speaker for c in self.cues if c.speaker})

    def to_prompt_text(self) -> str:
        lines = []
        for cue in self.cues:
            prefix = f"{cue.speaker}: " if cue.speaker else ""
            lines.append(f"[{format_timestamp(cue.start)}] {prefix}{cue.text}")
        return "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _to_seconds(stamp: str) -> float:
    parts = stamp.replace(",", ".").split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def _split_speaker(line: str) -> tuple[str | None, str]:
    speaker, sep, text = line.partition(": ")
    if sep and 0 < len(speaker) <= MAX_SPEAKER_LEN:
        return speaker.strip(), text.strip()
    return None, line.strip()


def parse_vtt(raw: str) -> Transcript:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        timing_idx = next((i for i, l in enumerate(lines) if TIMING_RE.match(l)), None)
        if timing_idx is None:
            continue
        start, end = TIMING_RE.match(lines[timing_idx]).groups()
        body = TAG_RE.sub("", " ".join(lines[timing_idx + 1 :])).strip()
        if not body:
            continue
        speaker, text = _split_speaker(body)
        cues.append(Cue(_to_seconds(start), _to_seconds(end), speaker, text))
    return Transcript(cues)


def parse_plain_text(raw: str) -> Transcript:
    cues = []
    for line in raw.splitlines():
        if line.strip():
            speaker, text = _split_speaker(line)
            cues.append(Cue(0.0, 0.0, speaker, text))
    return Transcript(cues)


def load_transcript_text(raw: str) -> Transcript:
    transcript = parse_vtt(raw)
    return transcript if transcript.cues else parse_plain_text(raw)


def load_transcript_file(path: str | Path) -> Transcript:
    return load_transcript_text(Path(path).read_text(encoding="utf-8", errors="replace"))
