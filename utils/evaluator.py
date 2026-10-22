"""Gemini evaluation over redacted text only: transcript + local features + classifier output.

No frames or audio are sent; visual and audio evidence reaches Gemini as numeric features.
"""

from __future__ import annotations

import os
import time
from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from utils.features import format_features
from utils.transcript_processor import Transcript, format_timestamp

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_RETRIES = 4
RETRYABLE_CODES = {429, 500, 503}


class TimestampedObservation(BaseModel):
    timestamp: str
    source: Literal["transcript", "visual", "audio"]
    observation: str


class Evaluation(BaseModel):
    predicted_category: Literal["poor", "average", "good", "excellent"]
    confidence_score: float
    quality_issue: bool
    quality_issue_details: str
    summary: str
    key_strengths: list[str]
    areas_for_improvement: list[str]
    timestamped_observations: list[TimestampedObservation]


# Rubric derived from the reviewer comments in feedbacks.pdf.
SYSTEM_PROMPT = """\
You are an experienced academic quality reviewer for 8x8, an online education company
that teaches small groups of children over video calls. You assess recorded classes
and give the tutor specific, evidence-based feedback.

You do not see the video. You receive:
- a transcript in which the tutor is labelled "Tutor", students "Student N", and other
  names are replaced with "[name]";
- visual and audio signals measured locally from the recording;
- possibly, estimates from small local classifiers trained on 10 labelled classes
  (only those that beat a majority baseline; treat them as weak evidence).

Evaluate the class against this rubric:

1. Opening & rapport building - warm greeting, small talk, gives new joiners time to
   introduce themselves; does NOT jump straight into content or revision.
2. Individual engagement - addresses students by name, asks each student questions
   individually, includes everyone (not just the most vocal child).
3. Tone & energy - energetic, pleasant, happy interaction suited to children.
4. Pedagogical clarity & pacing - clear explanations, sensible pace, checks for
   understanding, uses puzzles/examples/videos where helpful.
5. Visual presentation - tutor camera on and face visible, adequate lighting
   (judge from the visual signals).
6. Closure - recap/conclusion at the end and homework (HW) assigned.
7. Technical quality - audio/video clarity, freezes, silences (judge from the signals
   and from complaints in the transcript such as "can't hear you").

Category definitions:
- excellent: strong on nearly every rubric item, especially rapport, individual
  engagement and homework.
- good: solid class with a good introduction and inclusive questioning, but one or two
  clear gaps (e.g. no homework, interaction could be more pleasant).
- average: the basics are present but energy, clarity, lighting or homework are
  noticeably lacking.
- poor: weak opening/closing, little rapport, low energy, camera off or face not
  visible, or persistent technical quality problems.

Set `quality_issue` to true when technical/AV problems materially affect the class,
independently of the overall category.

Rules:
- Every strength, improvement and observation must cite concrete evidence: a quote
  with its timestamp, or a named signal with its value. Avoid generic statements.
- Refer to people only as "the tutor" or "Student N".
- Use "visual"/"audio" as the observation source only when citing those signals.
- `confidence_score` is between 0 and 1; lower it when evidence is limited.
"""


def get_client(api_key: str | None = None) -> genai.Client:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def build_contents(
    transcript: Transcript | None,
    features: dict[str, float],
    estimates: list[str],
    duration: float,
) -> list[str]:
    contents = [
        f"Class recording duration: {format_timestamp(duration)}.",
        "Signals measured locally from the recording:\n" + format_features(features),
    ]
    if estimates:
        contents.append("Local classifier estimates:\n" + "\n".join(f"- {e}" for e in estimates))
    if transcript and transcript.cues:
        contents.append("Redacted transcript ([mm:ss] from start of recording):\n" + transcript.to_prompt_text())
    else:
        contents.append("No transcript is available; rely on the signals and lower `confidence_score`.")
    contents.append("Now evaluate this class using the rubric and return the JSON result.")
    return contents


def evaluate_class(client: genai.Client, contents: list[str], model: str = DEFAULT_MODEL) -> Evaluation:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=Evaluation,
        temperature=0.2,
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            break
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_CODES or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt * 10)

    evaluation = response.parsed or Evaluation.model_validate_json(response.text)
    evaluation.confidence_score = min(max(evaluation.confidence_score, 0.0), 1.0)
    return evaluation
