# Find My Trial

Find My Trial is now structured as a real chart-to-trial ranking workspace instead of a single inline Flask page.

## What Changed

- Rebuilt patient parsing to handle messy chart text, structured JSON/CSV rows, and RTF-like exports.
- Added a hybrid ranking engine that blends:
  - word-level TF-IDF relevance
  - character-level TF-IDF similarity
  - optional BioBERT + ClinicalBERT reranking from local cache
  - structured overlap on diagnoses, biomarkers, therapies, demographics, status, and location
- Replaced the inline HTML with a branded UI that surfaces:
  - patient summary and extracted clinical signals
  - ranked trial cards
  - explicit match reasons and cautions
  - per-result score breakdowns
- Added tests for patient parsing and ranking behavior.

## Run

```bash
cd Code
source .venv/bin/activate
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Tests

```bash
cd Code
source .venv/bin/activate
python -m unittest discover tests
```

## Notes

- Default data source: `SampleData.csv`
- Default cache directory: `.cache`
- Semantic reranking uses local model files only and gracefully falls back if they are unavailable.
- This version still ranks against the 10,000-row sample, so it is a discovery aid rather than a full eligibility screener.
