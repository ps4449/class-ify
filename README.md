# class-ify
An assignment for Humaniti AI recruitment: an app that rates online class recordings as **Poor / Average / Good / Excellent** and backs each rating with specific observations.

## Approach: hybrid, privacy-first
Everything that touches identifiable data runs **locally**. Gemini only ever receives redacted text. The app shows **exactly what was sent to Gemini** for every evaluation.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env      # add your GEMINI_API_KEY
python train.py           # local only: extracts features, trains, prints leave-one-out accuracy
streamlit run app.py
```

Put the dataset at `data/<category>/` alongside `data/feedbacks.pdf`. The folder is gitignored because the recordings are confidential. `models/classifier.joblib` holds only model coefficients, no recordings or text, so it can be committed for the hosted app.

## Limitations
- There are only 10 labelled recordings. Leave-one-out results: teaching rating 50% exact (baseline 25%) and 88% within one level, on 8 recordings; rapport 67% and homework 100%, on 6 labelled each. These are sent to Gemini. Energy, tutor camera, new-joiner intros and the AV-issue model don't beat their baselines, so they're shown locally only.
- The rubric cut-offs were set by hand after looking at this same data, so the teaching-rating accuracy is somewhat optimistic.
- Redaction uses speaker labels plus fuzzy name matching. Unusual spellings of names in the transcript can slip through.
- Face detection can't tell the tutor apart from students in gallery view.
- When there is no transcript, Whisper output has no speaker labels, so the speaker-based signals are missing.
