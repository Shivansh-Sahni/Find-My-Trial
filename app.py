from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from matching import TrialMatcher


BASE_DIR = Path(__file__).resolve().parent
TRIALS_CSV_PATH = Path(os.environ.get("TRIALS_CSV_PATH", BASE_DIR / "SampleData.csv"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", BASE_DIR / ".cache"))
USE_GPU = os.environ.get("USE_GPU", "0") == "1"
ENABLE_SEMANTIC = os.environ.get("ENABLE_SEMANTIC", "1") != "0"
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "12"))

app = Flask(__name__)
matcher = TrialMatcher.from_csv(
    TRIALS_CSV_PATH,
    cache_dir=CACHE_DIR,
    use_gpu=USE_GPU,
    enable_semantic=ENABLE_SEMANTIC,
)


@app.get("/")
def index():
    return render_template("index.html", bootstrap=matcher.app_meta())


@app.get("/api/sample-patient")
def sample_patient():
    sample_path = BASE_DIR / "Sample_Patient_Chart.txt"
    return jsonify({"text": sample_path.read_text(encoding="utf-8", errors="ignore")})


@app.post("/api/match")
def api_match():
    payload = request.get_json(silent=True) if request.is_json else request.form
    if payload is None:
        payload = {}

    patient_text = str(payload.get("patient_text", "")).strip()
    upload_text = _read_uploaded_text()
    combined_text = "\n\n".join(part for part in [patient_text, upload_text] if part).strip()

    if not combined_text:
        return jsonify({"error": "Paste chart text or upload a text-like chart file to run matching."}), 400

    top_k = _coerce_int(payload.get("top_k"), DEFAULT_TOP_K)
    top_k = max(1, min(top_k, 30))

    response = matcher.search(
        combined_text,
        top_k=top_k,
        location=str(payload.get("location", "")).strip(),
        mode=str(payload.get("mode", "balanced")).strip().lower() or "balanced",
        active_only=_coerce_bool(payload.get("active_only"), True),
        interventional_only=_coerce_bool(payload.get("interventional_only"), True),
    )
    return jsonify(response)


@app.get("/health")
def health():
    return jsonify({"ok": True, "trials_loaded": matcher.app_meta()["trial_count"]})


def _read_uploaded_text() -> str:
    uploaded = request.files.get("patient_file")
    if not uploaded or not uploaded.filename:
        return ""
    return uploaded.read().decode("utf-8", errors="ignore")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
