import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors

from utils.benchmark import compare, comparisons_table, confusion_matrix, evaluation_text, summarize
from utils.categories import LABEL_COLORS
from utils.classifier import QUALITY_ISSUE, TEACHING, load_bundle, reliable_parts
from utils.criteria import TEACHING_CRITERIA
from utils.dataset import Sample, discover_samples
from utils.downloader import download
from utils.evaluator import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MODEL,
    Evaluation,
    get_client,
    ModelsUnavailable,
    failure_reason,
    parse_model_list,
)
from utils.features import FEATURE_DESCRIPTIONS
from utils.feedback_parser import GroundTruth, parse_feedback_pdf, resolve_ground_truth
from utils.pipeline import PipelineResult, cache_key, load_cached, run_pipeline, save_cached
from utils.transcriber import DEFAULT_WHISPER_MODEL
from utils.transcript_processor import Transcript, load_transcript_file, load_transcript_text
from utils.video_processor import DEFAULT_MAX_FRAMES, DEFAULT_TARGET_INTERVAL_S

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
FEEDBACK_PDF = DATA_DIR / "feedbacks.pdf"
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm"]
TRANSCRIPT_TYPES = ["vtt", "txt"]

st.set_page_config(page_title="Class-ify", layout="wide")


def get_setting(name: str, default: str = "") -> str:
    """Streamlit secrets first (hosted), then environment / .env (local)."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets.toml
        value = None
    return value or os.getenv(name) or default


@st.cache_resource
def cached_client(api_key: str | None):
    return get_client(api_key)


@st.cache_resource
def cached_bundle() -> dict | None:
    return load_bundle()


@st.cache_data
def load_ground_truth(path: str) -> dict[str, list[GroundTruth]]:
    return parse_feedback_pdf(path) if Path(path).exists() else {}


@st.cache_data
def load_samples(path: str) -> list[Sample]:
    return discover_samples(path)


def category_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#888")
    return (
        f"<span style='background:{color};color:white;padding:6px 16px;"
        f"border-radius:999px;font-weight:600;font-size:1.1rem'>{label.upper()}</span>"
    )


def gemini_error_message(exc: Exception) -> str:
    if isinstance(exc, ModelsUnavailable):
        advice = (
            "The free daily quota is used up for every model; it resets daily."
            if exc.all_daily_quota
            else "Please try again in a few minutes."
        )
        details = " · ".join(f"`{m}`: {failure_reason(e)}" for m, e in exc.failures)
        return f"No Gemini model is available right now. {advice}\n\n{details}"
    if isinstance(exc, genai_errors.APIError):
        return f"Gemini error {exc.code}: {exc.message}"
    return str(exc)


# ---------- Sidebar ----------

api_key = get_setting("GEMINI_API_KEY")
model = get_setting("GEMINI_MODEL", DEFAULT_MODEL)
fallback_models = parse_model_list(get_setting("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS))
whisper_model = get_setting("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
bundle = cached_bundle()
with st.sidebar:
    st.header("Status")
    if api_key:
        st.success("Gemini API key loaded")
    else:
        st.error("GEMINI_API_KEY not set (.env or Streamlit secrets)")
    if bundle:
        reliable = reliable_parts(bundle)
        teaching = bundle["metrics"][TEACHING]
        status = (
            f"Local classifier ({bundle['n_samples']} samples): teaching rating LOOCV "
            f"{teaching['accuracy']:.0%} vs {teaching['baseline']:.0%} baseline. "
            f"{len(reliable)} of {len(bundle['metrics'])} parts beat their baseline and are sent to Gemini."
        )
        (st.success if reliable else st.warning)(status)
    else:
        st.warning("No local classifier found. Run `python train.py`.")
    st.markdown(
        f"- Gemini model: `{model}`\n"
        f"- Fallbacks if overloaded: {', '.join(f'`{m}`' for m in fallback_models) or 'none'}\n"
        f"- Whisper model (no transcript): `{whisper_model}`\n"
        f"- Frames: every {DEFAULT_TARGET_INTERVAL_S}s, max {DEFAULT_MAX_FRAMES} (fixed to match training)"
    )
    st.caption("Frames and audio stay local. Gemini receives only the redacted transcript and numeric signals.")

st.title("Class-ify")

ground_truth = load_ground_truth(str(FEEDBACK_PDF))
samples = load_samples(str(DATA_DIR))


def render_result(result: PipelineResult):
    evaluation, local = result.evaluation, result.local
    st.markdown(category_badge(evaluation.predicted_category), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{evaluation.confidence_score:.0%}")
    c2.metric("Duration", f"{local.duration / 60:.0f} min")
    c3.metric("Quality issue", "Yes" if evaluation.quality_issue else "No")
    c4.metric("Local teaching rating", result.classifier.teaching_label.title() if result.classifier else "—")
    st.caption(
        f"Evaluated with `{result.model_used}`"
        + (f" (fallback: `{model}` was overloaded)" if result.model_used != model else "")
    )
    if local.transcribed:
        st.info("No transcript supplied; transcribed locally with Whisper (no speaker labels).")
    if evaluation.quality_issue:
        st.warning(evaluation.quality_issue_details)

    st.subheader("Summary")
    st.write(evaluation.summary)

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Key strengths")
        for item in evaluation.key_strengths:
            st.markdown(f"- {item}")
    with right:
        st.subheader("🛠 Areas for improvement")
        for item in evaluation.areas_for_improvement:
            st.markdown(f"- {item}")

    st.subheader("Timestamped observations")
    st.dataframe(
        pd.DataFrame([o.model_dump() for o in evaluation.timestamped_observations]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Local signals"):
        st.dataframe(
            pd.DataFrame(
                [{"signal": k, "value": v, "meaning": FEATURE_DESCRIPTIONS[k]} for k, v in local.features.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if result.classifier:
            reliable = reliable_parts(bundle)
            st.markdown("**Local classifier estimates**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "estimate": c.description,
                            "probability": round(result.classifier.criteria[c.name], 2),
                            "sent_to_gemini": c.name in reliable,
                        }
                        for c in TEACHING_CRITERIA
                    ]
                    + [
                        {
                            "estimate": "Audio/video quality issue",
                            "probability": round(result.classifier.quality_issue, 2),
                            "sent_to_gemini": QUALITY_ISSUE in reliable,
                        },
                        {
                            "estimate": f"Teaching rating: {result.classifier.teaching_label}",
                            "probability": round(result.classifier.teaching_score, 2),
                            "sent_to_gemini": TEACHING in reliable,
                        },
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Only estimates whose leave-one-out accuracy beats the majority baseline are sent to Gemini.")

    with st.expander("Exactly what was sent to Gemini"):
        st.text(result.sent_to_gemini)

    if local.keyframes:
        with st.expander(f"Sampled keyframes ({len(local.keyframes)}, local only)"):
            cols = st.columns(4)
            for i, frame in enumerate(local.keyframes):
                cols[i % 4].image(frame.image, caption=f"{frame.timestamp / 60:.1f} min")


def render_ground_truth(video_id: str | None, video_name: str, evaluation: Evaluation):
    st.divider()
    st.subheader("Ground truth comparison")
    candidates = resolve_ground_truth(ground_truth, video_name)
    comparison = (
        compare(video_id, evaluation.predicted_category, candidates, evaluation_text(evaluation))
        if video_id
        else None
    )
    if not comparison:
        st.info(f"No human feedback found for video ID `{video_id}`.")
        return

    if comparison.ambiguous:
        st.warning(
            f"{video_id} has {len(candidates)} feedback entries and the file name date doesn't "
            "pick one; comparing against the closest."
        )
    ai_col, human_col = st.columns(2)
    with ai_col:
        st.markdown("**AI evaluation**")
        st.markdown(category_badge(comparison.predicted), unsafe_allow_html=True)
        st.write(evaluation.summary)
    with human_col:
        gt = comparison.ground_truth
        st.markdown(f"**Human feedback** ({gt.tutor}, {gt.date})")
        st.markdown(category_badge(gt.ground_truth_category), unsafe_allow_html=True)
        st.caption(f"Source category: {gt.source_category}")
        st.write(gt.ground_truth_feedback)

    st.metric("Category match", "✅ Yes" if comparison.category_match else f"❌ Off by {comparison.distance}")
    st.write("**Themes raised by both:**", ", ".join(comparison.theme_overlap) or "—")
    missed = [t for t in comparison.themes_in_gt if t not in comparison.theme_overlap]
    st.write("**Themes the AI missed:**", ", ".join(missed) or "—")


def resolve_transcript(upload, url: str, workdir: Path) -> Transcript | None:
    if upload is not None:
        return load_transcript_text(upload.getvalue().decode("utf-8", errors="replace"))
    if url.strip():
        return load_transcript_file(download(url.strip(), workdir))
    return None


tab_eval, tab_batch = st.tabs(["Evaluate a recording", "Batch benchmark"])

# ---------- Single evaluation ----------

with tab_eval:
    source = st.radio("Recording source", ["Dataset sample", "Upload", "Link"], horizontal=True)

    sample = video_upload = None
    video_url = transcript_url = ""
    transcript_upload = None

    if source == "Dataset sample":
        if not samples:
            st.warning(f"No samples found in `{DATA_DIR}`.")
        else:
            sample = st.selectbox("Sample", samples, format_func=lambda s: s.label)
            if sample and not sample.transcript_path:
                st.caption("No matching transcript found; Whisper will be used.")
            if bundle and sample and sample.key in bundle["loocv"]:
                st.caption("This sample was used for training; its leave-one-out prediction is used.")
    else:
        if source == "Upload":
            video_upload = st.file_uploader("Class recording", type=VIDEO_TYPES)
        else:
            video_url = st.text_input("Recording link (Google Drive or direct URL)")
        st.markdown("**Transcript (optional)**")
        transcript_url = st.text_input("Transcript link")
        transcript_upload = st.file_uploader("…or upload transcript", type=TRANSCRIPT_TYPES)

    ready = bool(api_key) and bool(sample or video_upload or video_url.strip())
    if st.button("Evaluate", type="primary", disabled=not ready):
        client = cached_client(api_key)
        override = bundle["loocv"].get(sample.key) if bundle and sample else None
        with tempfile.TemporaryDirectory() as tmp, st.status("Running pipeline…") as status:
            workdir = Path(tmp)
            try:
                if sample:
                    video_path, video_name = sample.video_path, sample.video_path.name
                    transcript = load_transcript_file(sample.transcript_path) if sample.transcript_path else None
                elif video_upload:
                    video_path = workdir / video_upload.name
                    video_path.write_bytes(video_upload.getvalue())
                    video_name = video_upload.name
                    transcript = resolve_transcript(transcript_upload, transcript_url, workdir)
                else:
                    status.update(label="Downloading recording…")
                    video_path = download(video_url.strip(), workdir)
                    video_name = video_path.name
                    transcript = resolve_transcript(transcript_upload, transcript_url, workdir)

                st.session_state["result"] = run_pipeline(
                    client,
                    video_path,
                    transcript,
                    model=model,
                    fallback_models=fallback_models,
                    classifier_bundle=bundle,
                    classifier_override=override,
                    video_name=video_name,
                    on_step=lambda msg: status.update(label=msg),
                    whisper_model=whisper_model,
                )
                status.update(label="Done", state="complete")
            except (ModelsUnavailable, genai_errors.APIError) as exc:
                status.update(label="Failed", state="error")
                st.error(gemini_error_message(exc))
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.exception(exc)

    result: PipelineResult | None = st.session_state.get("result")
    if result:
        render_result(result)
        render_ground_truth(result.video_id, result.video_name, result.evaluation)

# ---------- Batch benchmark ----------

with tab_batch:
    labelled = [s for s in samples if s.video_id in ground_truth]
    if not samples:
        st.info(f"No dataset found in `{DATA_DIR}`. The batch benchmark runs locally only.")
    selected = st.multiselect("Recordings", samples, default=labelled, format_func=lambda s: s.label)
    with st.expander("Advanced"):
        force = st.checkbox("Ignore cached results")
        rpm = st.number_input("Gemini requests per minute (free tier allows 15)", 1, 15, 10)
    st.caption("Training samples use leave-one-out classifier predictions, so Gemini never sees an in-sample prior.")

    if st.button("Run batch", type="primary", disabled=not (api_key and selected)):
        client = cached_client(api_key)
        progress = st.progress(0.0)
        rows: list[tuple[Sample, Evaluation, str | None, str]] = []
        for i, s in enumerate(selected):
            progress.progress(i / len(selected), text=f"Evaluating {s.label}")
            override = bundle["loocv"].get(s.key) if bundle else None
            key = cache_key(
                s.key, model, DEFAULT_TARGET_INTERVAL_S, DEFAULT_MAX_FRAMES, bundle["trained_at"] if bundle else None
            )
            cached = None if force else load_cached(key)
            if cached:
                evaluation, model_used = cached
            else:
                try:
                    transcript = load_transcript_file(s.transcript_path) if s.transcript_path else None
                    result = run_pipeline(
                        client,
                        s.video_path,
                        transcript,
                        model=model,
                        fallback_models=fallback_models,
                        classifier_bundle=bundle,
                        classifier_override=override,
                        whisper_model=whisper_model,
                    )
                except Exception as exc:
                    message = gemini_error_message(exc)
                    st.error(f"{s.label}: {message}")
                    continue
                evaluation, model_used = result.evaluation, result.model_used
                override = override or result.classifier
                save_cached(key, evaluation, model_used)
                time.sleep(60 / rpm)
            rows.append((s, evaluation, override.teaching_label if override else None, model_used))
        progress.progress(1.0, text="Done")
        st.session_state["batch"] = rows

    rows = st.session_state.get("batch", [])
    if rows:
        st.subheader("Predictions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "recording": s.label,
                        "video_id": s.video_id,
                        "gemini": e.predicted_category,
                        "classifier": clf,
                        "confidence": round(e.confidence_score, 2),
                        "quality_issue": e.quality_issue,
                        "model": used,
                    }
                    for s, e, clf, used in rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if any(used != model for *_, used in rows):
            st.caption(f"Some recordings were evaluated with a fallback model because `{model}` was overloaded.")

        def candidates_for(s: Sample) -> list[GroundTruth]:
            return resolve_ground_truth(ground_truth, s.video_path.name)

        gemini_cmp = [
            c
            for s, e, _, _ in rows
            if s.video_id
            and (c := compare(s.video_id, e.predicted_category, candidates_for(s), evaluation_text(e)))
        ]
        # The classifier rates teaching only, so AV quality-issue recordings are left out.
        classifier_cmp = [
            c
            for s, _, clf, _ in rows
            if s.video_id
            and clf
            and (c := compare(s.video_id, clf, candidates_for(s)))
            and c.ground_truth.source_category != "quality-issue"
        ]
        stats = summarize(gemini_cmp)
        if stats["n"]:
            st.subheader("Against ground truth")
            clf_stats = summarize(classifier_cmp)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gemini exact accuracy", f"{stats['accuracy']:.0%}")
            m2.metric("Gemini within one level", f"{stats['within_one']:.0%}")
            m3.metric("Classifier teaching accuracy", f"{clf_stats['accuracy']:.0%}" if clf_stats["n"] else "—")
            m4.metric(
                "Theme recall",
                f"{stats['theme_recall']:.0%}" if stats["theme_recall"] is not None else "—",
            )
            if stats["ambiguous"]:
                st.caption(f"{stats['ambiguous']} recording(s) matched several feedback entries (no date match).")
            st.dataframe(comparisons_table(gemini_cmp), use_container_width=True, hide_index=True)
            st.markdown("**Gemini confusion matrix** (rows = ground truth, columns = predicted)")
            st.dataframe(confusion_matrix(gemini_cmp), use_container_width=True)
