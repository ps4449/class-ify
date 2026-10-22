# Implementation Guide: 8x8 Class Quality Evaluator AI

## 1. Project Overview & Architecture

This application evaluates the quality of online educational recordings using a **Hybrid Privacy-First Pipeline**: all raw media is processed locally, and **Gemini (Free Tier)** receives only redacted text and numeric signals. The UI is built with **Streamlit**.

### Target Quality Categories
- Output labels: `poor`, `average`, `good`, `excellent`
- Source category mapping: `weak`, `quality-issue` → `poor`; `average`, `good`, `excellent` unchanged; `demo` has no ground truth
- Separate boolean `quality_issue` flags audio/video problems

### Pipeline Flow
```
Video File / Link + optional Transcript (upload / link)
       │
       ├──> [local] Keyframe sampling (OpenCV)
       ├──> [local] Transcript parsing (.vtt) or Whisper transcription
       ├──> [local] Name redaction → "Tutor" / "Student N" / "[name]"
       ├──> [local] Feature extraction (transcript, visual, audio)
       ├──> [local] Criterion + AV-issue classifiers (scikit-learn) → rubric teaching rating
       │
       ▼
[Gemini] Redacted transcript + features + rubric (+ classifier estimates that beat baseline)
       │
       ▼
JSON Evaluation Result (Category + Structured Observations)
       │
       ├──> Display in Streamlit Interactive Dashboard
       └──> Compare against Ground Truth (Extracted from feedbacks.pdf)
```

---

## 2. Environment Setup & Dependencies

`requirements.txt`:

```text
streamlit>=1.30.0
google-genai>=1.0.0
pydantic>=2.0
opencv-python-headless
faster-whisper>=1.0.0
scikit-learn>=1.3
joblib
numpy
pypdf
pandas
pillow
python-dotenv
requests
gdown
```

`.env` (or Streamlit secrets):
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite
WHISPER_MODEL=small
DATA_DIR=data
```

`.streamlit/config.toml`: `maxUploadSize = 1024` (recordings exceed the 200 MB default).

---

## 3. Ordered Implementation Subtasks

### Subtask 1: PDF Feedback Extractor (`utils/feedback_parser.py`)
- Parse `feedbacks.pdf` using `pypdf`; split by category section headers.
- Extract entries: batch ID, date, tutor, feedback text.
- Match video IDs by batch ID at the start of the file name (e.g. `B435 Intermediate ...` → `B435`).
- Lookup: `{ video_id: [ { source_category, ground_truth_category, ground_truth_feedback, date, parsed_date, tutor } ] }` (a batch ID may have multiple entries).
- Resolution: batch ID is the primary key; only when an ID has several entries, select the entry whose date equals the `YYYY-MM-DD` date in the file name. If no date matches, keep all candidates.

### Subtask 2: Dataset Discovery (`utils/dataset.py`)
- Scan `data/<category>/` for videos and transcripts; strip trailing spaces from folder names.
- Pair each video with the transcript in the same folder sharing the most name tokens.

### Subtask 3: Frame Sampling (`utils/video_processor.py`)
- Extract keyframes with `cv2.VideoCapture` every 15–30 s; stretch the interval so at most `max_frames` are kept.
- Convert to RGB `PIL.Image`, resized to max width 768. Frames are never sent off-device.

### Subtask 4: Transcript Loader (`utils/transcript_processor.py`, `utils/transcriber.py`)
- Parse `.vtt` cues into `(start, end, speaker, text)`; fall back to plain text.
- If no transcript is supplied, transcribe locally with `faster-whisper` (default `small`, `int8`, VAD filter). Output has no speaker labels.

### Subtask 5: Redaction (`utils/redaction.py`)
- Tutor = speaker with the most talk time → `Tutor`; other speakers → `Student N` by first appearance.
- Spoken names: fuzzy-match capitalized words (ratio ≥ 0.75) against speaker-label tokens, excluding device words and a common-word stoplist.
- Names following greetings (`hi`, `hello`, `welcome`, `good evening`, ...) → `[name]`.

### Subtask 6: Feature Extraction (`utils/features.py`)
| Group | Features |
|---|---|
| Transcript | tutor talk ratio, students speaking, student turn balance, share of students addressed by name, tutor questions/min, student turns/min, rapport cues in first 5 min, new-joiner intro, homework mentioned (whole class), conclusion cues in last 10%, gaps > 60 s per hour |
| Visual | tutor face ratio (Haar cascade, frames during tutor speech only), mean brightness, dark-frame ratio, near-blank frame ratio, log sharpness (Laplacian variance) |
| Audio | silence ratio, tutor loudness median and std (1 s RMS windows during tutor cues, 16 kHz) |

- Tutor-only measurements use tutor cue timings from the transcript; without speaker labels they fall back to all speech.
- Missing values are NaN.

### Subtask 7: Criterion Labels (`utils/criteria.py`)
- Teaching criteria from the reviewer feedback, each mapped to 1–2 signals:

| Criterion | Signals | Unmentioned |
|---|---|---|
| `rapport_intro` | rapport cues in opening, tutor talk ratio | unknown |
| `homework` | homework mentioned | unknown |
| `energetic_tone` | tutor loudness std, student turns/min | met |
| `tutor_camera` | tutor face ratio, near-blank frame ratio | met |
| `new_joiner_intro` | new-joiner intro | met |

- Labels parsed from each `feedbacks.pdf` entry: split on commas and "but"; a clause with a criterion keyword is criticised (0) if it contains need / should / without / can't / can / no / not / turned off, else met (1). Any criticism wins.
- "Unmentioned" policy: criteria reviewers raise only when lacking default to met; others are unknown and excluded from training.

### Subtask 8: Local Classifier (`utils/classifier.py`, `train.py`)
- One binary model per criterion: median imputation → standard scaling → logistic regression (`C=1.0`, `class_weight="balanced"`), trained on its own signals.
- Teaching rating: mean criterion probability mapped by fixed cut-offs (≥ 0.9 excellent, ≥ 0.7 good, ≥ 0.45 average, else poor); not fitted.
- AV quality issue: separate binary model on dark-frame ratio, near-blank frame ratio, sharpness, tutor face ratio, silence ratio. Labels from the `quality-issue` folder.
- Teaching rating target: folder label, excluding `quality-issue` recordings.
- Validation: leave-one-out cross-validation per criterion, teaching rating and AV issue; LOOCV predictions stored in the model bundle.
- Training samples use their LOOCV prediction at inference to avoid leakage.
- Reliability gate: each estimate (criterion, teaching rating, AV issue) is sent to Gemini only if its LOOCV accuracy exceeds its majority-class baseline; otherwise it is displayed locally only.
- Artifact: `models/classifier.joblib` (coefficients only, no media or text).

### Subtask 9: Gemini Evaluation Engine (`utils/evaluator.py`)
- `google-genai` client; model from `GEMINI_MODEL`.
- Inputs: redacted transcript, formatted feature values, classifier estimates that pass the reliability gate. No frames or audio.
- Rubric: opening & rapport, individual engagement, tone & energy, clarity & pacing, visual presentation, closure & homework, technical quality.
- Strict JSON via response schema:
  - `predicted_category`
  - `confidence_score`
  - `quality_issue`, `quality_issue_details`
  - `summary`
  - `key_strengths`
  - `areas_for_improvement`
  - `timestamped_observations` (`timestamp`, `source`: transcript / visual / audio, `observation`)
- Every claim must cite a timestamped quote or a named signal value.

### Subtask 10: Benchmarking & Comparison Logic (`utils/benchmark.py`)
- Match predictions to ground truth by batch ID, disambiguated by file name date (Subtask 1); if several entries still remain, flag as ambiguous and compare against the closest entry.
- Metrics: exact accuracy, within-one-level accuracy, confusion matrix, theme recall (reviewer themes also raised by the AI) for Gemini; teaching-rating accuracy for the classifier, excluding `quality-issue` recordings.

### Subtask 11: Pipeline Orchestration (`utils/pipeline.py`)
- `analyze_locally`: keyframes → transcript/Whisper → redaction → features.
- `run_pipeline`: local analysis → classifier → Gemini; returns the evaluation and the exact text sent to Gemini.
- Disk cache: `.cache/features/`, `.cache/evaluations/`.

### Subtask 12: Streamlit Web UI (`app.py`)
- **Header & Status (read-only):** API key status, classifier status (teaching LOOCV vs baseline, parts sent to Gemini), active Gemini and Whisper models, fixed frame settings.
- **Settings:** `GEMINI_API_KEY`, `GEMINI_MODEL`, `WHISPER_MODEL` read from Streamlit secrets, then environment / `.env`. Frame interval and max frames are constants shared with `train.py` (changing them requires retraining). Not editable in the UI.
- **Input:** dataset sample, video upload, or video link (Google Drive / direct URL); transcript optional via upload or link.
- **Analysis View:** status steps, category badge, confidence, quality-issue flag, strengths, improvements, timestamped observations, local signals, classifier estimates (criteria, AV issue, teaching rating) with sent-to-Gemini status, "Exactly what was sent to Gemini" panel, keyframes (local only).
- **Ground Truth Comparison View:** side-by-side AI vs human feedback, category match, shared and missed themes.
- **Batch Evaluation Dashboard (local only, needs `data/`):** run across dataset samples; requests-per-minute and cache override under "Advanced"; predictions table, accuracy metrics, confusion matrix.

---

## 4. Operational Guidelines

### Privacy
1. Frames, audio and unredacted transcripts never leave the machine.
2. Gemini receives only redacted text and numeric signals.
3. `data/`, `.env` and `.cache/` are gitignored.

### Gemini Free Tier (15 RPM)
1. Pace batch requests (configurable RPM) and retry 429/5xx with exponential backoff.
2. On 500/503 (overloaded) or 429 (quota; daily quota is skipped without retrying), fall back through `GEMINI_FALLBACK_MODELS`; the model actually used is shown per evaluation and stored in the cache.
3. Cache evaluations keyed by sample, model, frame settings and classifier version.
