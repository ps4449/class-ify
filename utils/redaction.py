"""Replace speaker and spoken names with "Tutor" / "Student N" before anything leaves the machine."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache

from utils.transcript_processor import Cue, Transcript

TUTOR = "Tutor"
STUDENT_PREFIX = "Student"
NAME_PLACEHOLDER = "[name]"
MATCH_THRESHOLD = 0.75

WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
GREETING_RE = re.compile(
    r"\b((?i:hi|hello|hey|bye|welcome|thanks|thank you|well done|very good|good job"
    r"|good (?:morning|afternoon|evening|night)))([,!.]?\s+)([A-Z][a-z]{2,})\b"
)
DEVICE_WORDS = {
    "iphone", "ipad", "android", "galaxy", "phone", "laptop", "desktop", "samsung",
    "redmi", "oppo", "vivo", "realme", "oneplus", "pixel", "mom", "dad", "home",
}
COMMON_WORDS = {
    "okay", "yes", "yeah", "the", "and", "but", "now", "then", "today", "tomorrow",
    "good", "great", "very", "what", "which", "where", "when", "why", "how", "who",
    "can", "could", "will", "would", "should", "this", "that", "there", "here",
    "hello", "hey", "bye", "sir", "mam", "maam", "madam", "miss", "teacher",
    "everyone", "everybody", "all", "guys", "friends", "class", "student", "tutor",
    "name", "please", "thank", "thanks", "sorry", "wait", "let", "lets", "come",
    "see", "look", "right", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "english", "maths", "math", "chess", "king", "queen",
    "rook", "bishop", "knight", "pawn", "white", "black", "move", "check",
}


def _name_tokens(label: str) -> list[str]:
    tokens = (t.lower() for t in re.findall(r"[A-Za-z]+", label))
    return [t for t in tokens if len(t) >= 3 and t not in DEVICE_WORDS and t not in COMMON_WORDS]


def _assign_aliases(cues: list[Cue]) -> dict[str, str]:
    talk_time: dict[str, float] = defaultdict(float)
    order: list[str] = []
    for cue in cues:
        if cue.speaker:
            if cue.speaker not in talk_time:
                order.append(cue.speaker)
            talk_time[cue.speaker] += cue.end - cue.start
    if not order:
        return {}
    # The tutor is assumed to be the speaker with the most talk time.
    tutor = max(talk_time, key=talk_time.get)
    aliases = {tutor: TUTOR}
    for speaker in order:
        if speaker != tutor:
            aliases[speaker] = f"{STUDENT_PREFIX} {len(aliases)}"
    return aliases


def redact_transcript(transcript: Transcript) -> Transcript:
    aliases = _assign_aliases(transcript.cues)
    name_index = [(token, alias) for label, alias in aliases.items() for token in _name_tokens(label)]

    @lru_cache(maxsize=None)
    def resolve(word: str) -> str | None:
        lower = word.lower()
        if lower in COMMON_WORDS:
            return None
        best_alias, best_ratio = None, 0.0
        for token, alias in name_index:
            ratio = SequenceMatcher(None, lower, token).ratio()
            if ratio > best_ratio:
                best_alias, best_ratio = alias, ratio
        return best_alias if best_ratio >= MATCH_THRESHOLD else None

    def redact_text(text: str) -> str:
        text = WORD_RE.sub(lambda m: resolve(m.group()) or m.group(), text)
        return GREETING_RE.sub(
            lambda m: m.group(0)
            if m.group(3).lower() in COMMON_WORDS
            else f"{m.group(1)}{m.group(2)}{NAME_PLACEHOLDER}",
            text,
        )

    return Transcript(
        [
            Cue(c.start, c.end, aliases.get(c.speaker) if c.speaker else None, redact_text(c.text))
            for c in transcript.cues
        ]
    )
