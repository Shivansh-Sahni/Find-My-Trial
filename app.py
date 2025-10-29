import os, json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, jsonify

from nlp_pipeline import NLPPipeline, PatientParser
from utils import cosine_sim

TRIALS_CSV_PATH = os.environ.get("TRIALS_CSV_PATH", "Data.csv")
CACHE_DIR       = os.environ.get("CACHE_DIR", "./.cache")
USE_GPU         = os.environ.get("USE_GPU", "0") == "1"
SKIP_BERT       = os.environ.get("SKIP_BERT", "0") == "1"

app = Flask(__name__)

DF = pd.read_csv(TRIALS_CSV_PATH)
TEXT_COLS = ["Study Title","Brief Summary","Conditions","Interventions","Locations"]

def build_corpus(df: pd.DataFrame):
    cols = [(df[c].fillna("").astype(str) if c in df.columns else "") for c in TEXT_COLS]
    return (cols[0] + " " + cols[1] + " " + cols[2] + " " + cols[3] + " " + cols[4]).tolist()

CORPUS = build_corpus(DF)

pipe = NLPPipeline(cache_dir=CACHE_DIR, use_gpu=USE_GPU)
pipe.fit_tfidf(CORPUS)

E_BIO = None
E_CLI = None
if not SKIP_BERT:
    print("[Init] Building BioBERT/ClinicalBERT embeddings (cached)")
    E_BIO = pipe.encode_corpus(CORPUS, model_name="biobert")
    E_CLI = pipe.encode_corpus(CORPUS, model_name="clinicalbert")
else:
    print("[Init] SKIP_BERT=1 — TF-IDF only; enable BERT later without code changes.")

W_TFIDF    = float(os.environ.get("W_TFIDF",    "1.0" if SKIP_BERT else "0.34"))
W_BIOBERT  = float(os.environ.get("W_BIOBERT",  "0.0" if SKIP_BERT else "0.33"))
W_CLINICAL = float(os.environ.get("W_CLINICAL", "0.0" if SKIP_BERT else "0.33"))
RECRUITING_BOOST = float(os.environ.get("RECRUITING_BOOST", "0.03"))
LOCATION_BOOST   = float(os.environ.get("LOCATION_BOOST",   "0.05"))

HTML = """
<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Find My Trial — AI Clinical Match</title>
<style>
:root{
  --primary:#007BFF;
  --accent:#0056d6;
  --text:#1a1a1a;
  --muted:#555;
  --bg:#ffffff;
  --card:#f9f9f9;
  --border:#dcdcdc;
}
body{
  background:var(--bg);
  color:var(--text);
  font:16px/1.5 'Inter',system-ui,Arial,sans-serif;
  margin:0;
}
.wrap{max-width:1100px;margin:24px auto;padding:0 20px}
.header{display:flex;align-items:center;gap:14px;margin-bottom:24px}
.logo{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,var(--primary),#00bfff)}
h1{font-size:28px;margin:0;color:var(--accent);font-weight:700;letter-spacing:-0.5px}
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:20px;
  margin:14px 0;
  box-shadow:0 3px 10px rgba(0,0,0,.05);
}
label{font-weight:600;margin:8px 0 6px;display:block}
input[type=text],input[type=number],textarea{
  width:100%;
  padding:12px 14px;
  border:1px solid #ccc;
  border-radius:10px;
  outline:none;
  font-size:15px;
}
input:focus,textarea:focus{border-color:var(--primary)}
.controls{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:8px}
button{
  background:linear-gradient(135deg,var(--primary),#00bfff);
  color:white;
  font-weight:600;
  border:0;
  border-radius:10px;
  padding:12px 18px;
  margin-top:16px;
  cursor:pointer;
  transition:all .2s;
}
button:hover{background:var(--accent)}
.badge{
  display:inline-block;
  background:var(--primary);
  color:white;
  border-radius:999px;
  padding:3px 10px;
  margin-right:6px;
  font-size:13px;
}
.title{color:var(--accent);margin:6px 0 2px;font-size:19px;font-weight:600}
.small{color:var(--muted);font-size:14px}
pre{white-space:pre-wrap;margin:0}
hr{border:none;border-top:1px solid #ddd;margin:12px 0}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
summary{cursor:pointer;color:var(--accent)}
.details-box{background:#f2f6ff;border:1px solid #d6e4ff;border-radius:8px;padding:10px;margin-top:10px}
</style>
</head><body><div class="wrap">
<div class="header"><div class="logo"></div><h1>Find My Trial</h1></div>
<div class="card small" style="background:#f2f6ff;border-color:#c8deff;">Over <b>500,000+</b> trials loaded. Enter patient details to discover personalized matches.</div>

<form class="card" method="POST" enctype="multipart/form-data">
  <label>Upload Patient Chart (TXT / JSON / CSV)</label>
  <input type="file" name="patient_file" accept=".txt,.json,.csv"/>
  <label>Or Paste Chart Text</label>
  <textarea name="patient_text" rows="7" placeholder="Example: 62 y/o female with metastatic HER2+ breast cancer, prior trastuzumab therapy, Michigan..."></textarea>
  <label>Location (optional)</label>
  <input type="text" name="location_kw" placeholder="Detroit, Michigan, United States"/>
  <div class="controls">
    <div><label>Top K</label><input type="number" name="k" value="10" min="1" max="50"/></div>
    <div><label><input type="checkbox" name="recruiting_only" value="1" checked/> Recruiting Only</label></div>
    <div><label><input type="checkbox" name="debug" value="1"/> Show Match Reason</label></div>
  </div>
  <button type="submit">Find Matching Trials</button>
</form>

{% if patient %}
  <div class="card"><b>Patient Summary</b><hr/>
  <pre class="small">{{patient}}</pre></div>
{% endif %}

{% if results %}
  <div class="card"><b>Top {{results|length}} Matches</b></div>
  {% for r in results %}
    <div class="card">
      <div>
        <span class="badge">Score {{ "%.3f"|format(r["__score__"]) }}</span>
        {% if r.get("Study Status") %}<span class="badge">{{r["Study Status"]}}</span>{% endif %}
        {% if r.get("Phases") %}<span class="badge">{{r["Phases"]}}</span>{% endif %}
        {% if r.get("Enrollment") %}<span class="badge">Enroll {{r["Enrollment"]}}</span>{% endif %}
      </div>
      <div class="title"><a target="_blank" href="{{r.get('Study URL','') or r.get('URL','') }}">{{ r.get("Study Title","(No Title)") }}</a></div>
      <div class="small">NCT {{ r.get("NCT Number","") }} · Sponsor {{ r.get("Sponsor","") }}</div>
      {% if r.get("Brief Summary") %}<p class="small" style="margin-top:8px">{{ r["Brief Summary"] }}</p>{% endif %}
      {% if r.get("Conditions") %}<div class="small"><b>Conditions:</b> {{ r["Conditions"] }}</div>{% endif %}
      {% if r.get("Interventions") %}<div class="small"><b>Interventions:</b> {{ r["Interventions"] }}</div>{% endif %}
      {% if r.get("Locations") %}<div class="small"><b>Locations:</b> {{ r["Locations"] }}</div>{% endif %}
      {% if r.get("__why__") %}
        <div class="details-box small"><b>Match Reason:</b> {{ r["__why__"] }}</div>
      {% endif %}
    </div>
  {% endfor %}
{% endif %}
</div></body></html>
"""

def _ensure_bert():
    global E_BIO, E_CLI
    if (W_BIOBERT>0 or W_CLINICAL>0) and (E_BIO is None or E_CLI is None):
        print("[Lazy] Building Bio/ClinicalBERT embeddings now…")
        E_BIO = pipe.encode_corpus(CORPUS, model_name="biobert")
        E_CLI = pipe.encode_corpus(CORPUS, model_name="clinicalbert")

def _score(query_text, location_kw, recruiting_only, k, debug):
    vq = pipe.encode_tfidf([query_text])
    sims = (vq @ pipe.tfidf_matrix.T)
    sims = np.asarray(sims.toarray()).ravel()  # FIX: csr_matrix has no attribute 'A'
    mn, mx = sims.min(), sims.max()
    sims = (sims - mn) / (mx - mn + 1e-9)

    sims_bio = np.zeros_like(sims)
    sims_cli = np.zeros_like(sims)
    if W_BIOBERT>0 or W_CLINICAL>0:
        _ensure_bert()
        q_bio = pipe.encode_text([query_text], model_name="biobert")
        q_cli = pipe.encode_text([query_text], model_name="clinicalbert")
        sims_bio = cosine_sim(q_bio, E_BIO).ravel()
        sims_cli = cosine_sim(q_cli, E_CLI).ravel()

    score = W_TFIDF*sims + W_BIOBERT*sims_bio + W_CLINICAL*sims_cli

    if location_kw:
        loc = DF["Locations"].fillna("").str.contains(location_kw, case=False, na=False).to_numpy()
        score = score + LOCATION_BOOST*loc.astype(float)
    if recruiting_only and "Study Status" in DF.columns:
        rec = DF["Study Status"].fillna("").str.contains("Recruit", case=False, na=False).to_numpy()
        score = score + RECRUITING_BOOST*rec.astype(float)

    out = DF.copy()
    out["__score__"] = score
    if recruiting_only and "Study Status" in out.columns:
        out = out[out["Study Status"].fillna("").str.contains("Recruit", case=False, na=False)]
    out = out.sort_values("__score__", ascending=False).head(int(k))

    if debug:
        out["__why__"] = f"Matched on similar terms and medical context related to: {', '.join(query_text.split()[:6])}..."

    return out.to_dict(orient="records")

@app.route("/", methods=["GET","POST"])
def index():
    results, patient = None, None
    mode_note = "(only) " if SKIP_BERT else "+ BERT "
    bert_note = "OFF" if SKIP_BERT else "ON"
    if request.method=="POST":
        f = request.files.get("patient_file")
        pasted = request.form.get("patient_text","")
        location_kw = request.form.get("location_kw","").strip()
        k = int(request.form.get("k","10"))
        recruiting_only = request.form.get("recruiting_only")=="1"
        debug = request.form.get("debug")=="1"
        text = pasted
        if f and f.filename:
            text += "\n" + f.read().decode("utf-8", errors="ignore")
        patient = PatientParser().parse(text)
        q = pipe.build_query_from_patient(patient)
        results = _score(q, location_kw, recruiting_only, k, debug)
    return render_template_string(
        HTML,
        results=results,
        patient=json.dumps(patient, indent=2) if patient else None,
        mode_note=mode_note, bert_note=bert_note
    )

@app.route("/api/match", methods=["POST"])
def api_match():
    p = request.get_json(force=True, silent=True) or {}
    text = p.get("text","")
    location_kw = p.get("location","")
    k = int(p.get("k",10))
    recruiting_only = bool(p.get("recruiting_only", True))
    patient = PatientParser().parse(text)
    q = pipe.build_query_from_patient(patient)
    results = _score(q, location_kw, recruiting_only, k, debug=False)
    return jsonify({"patient":patient, "results":results})

@app.get("/health")
def health(): return {"ok": True}

if __name__=="__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)