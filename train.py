"""Train the local criterion classifiers on data/<category>/ recordings.

Criterion labels are parsed from feedbacks.pdf; the teaching rating target is the
folder label, excluding quality-issue recordings, which train the separate AV model.
Runs entirely locally; nothing is sent to Gemini.

    python train.py [--data-dir data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.classifier import MODEL_PATH, TEACHING, is_reliable, save_bundle, train
from utils.criteria import TEACHING_CRITERIA, criterion_labels
from utils.dataset import discover_samples
from utils.feedback_parser import parse_feedback_pdf, resolve_ground_truth
from utils.pipeline import FEATURES_VERSION, analyze_locally, cache_key, load_cached_features, save_cached_features
from utils.transcript_processor import load_transcript_file
from utils.video_processor import DEFAULT_MAX_FRAMES, DEFAULT_TARGET_INTERVAL_S

QUALITY_ISSUE_FOLDER = "quality-issue"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-interval", type=float, default=DEFAULT_TARGET_INTERVAL_S)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    ground_truth = parse_feedback_pdf(args.data_dir / "feedbacks.pdf")
    samples = [s for s in discover_samples(args.data_dir) if s.quality_label]

    keys, rows, criteria, teaching, quality = [], [], [], [], []
    for sample in samples:
        key = cache_key(sample.key, args.target_interval, args.max_frames, FEATURES_VERSION)
        features = load_cached_features(key)
        if features is None:
            print(f"Extracting features: {sample.label}")
            transcript = load_transcript_file(sample.transcript_path) if sample.transcript_path else None
            features = analyze_locally(
                sample.video_path,
                transcript,
                target_interval_s=args.target_interval,
                max_frames=args.max_frames,
            ).features
            save_cached_features(key, features)

        entries = resolve_ground_truth(ground_truth, sample.video_path.name)
        is_quality_issue = sample.folder == QUALITY_ISSUE_FOLDER
        keys.append(sample.key)
        rows.append(features)
        criteria.append(
            criterion_labels(entries[0].ground_truth_feedback)
            if len(entries) == 1
            else {c.name: None for c in TEACHING_CRITERIA}
        )
        teaching.append(None if is_quality_issue else sample.quality_label)
        quality.append(int(is_quality_issue))

    bundle = train(keys, rows, criteria, teaching, quality)
    save_bundle(bundle, args.output)

    loocv = bundle["loocv"]
    print(
        pd.DataFrame(
            [
                {
                    "sample": k,
                    "rating": t or "(AV issue)",
                    "pred": loocv[k].teaching_label,
                    "score": round(loocv[k].teaching_score, 2),
                    "av_issue": q,
                    "p_av_issue": round(loocv[k].quality_issue, 2),
                }
                for k, t, q in zip(keys, teaching, quality)
            ]
        ).to_string(index=False)
    )
    print()
    print(
        pd.DataFrame(
            [
                {
                    "part": name,
                    "n": m["n"],
                    "loocv_acc": m["accuracy"],
                    "baseline": m["baseline"],
                    "sent_to_gemini": is_reliable(m),
                }
                for name, m in bundle["metrics"].items()
            ]
        ).to_string(index=False)
    )
    print(f"\nTeaching rating within one level: {bundle['metrics'][TEACHING]['within_one']}")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
