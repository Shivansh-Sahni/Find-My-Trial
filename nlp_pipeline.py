import os, json, re
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch, spacy, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

from utils import mean_pooling

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

NLTK_STOPS = set(nltk.corpus.stopwords.words("english"))

class NLPPipeline:
    def __init__(self, cache_dir="./.cache", use_gpu=False):
        self.cache = cache_dir; os.makedirs(self.cache, exist_ok=True)
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.vec: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._tok_bio=self._mdl_bio=self._tok_cli=self._mdl_cli=None

    def normalize(self, text: str) -> str:
        doc = self.nlp((text or "").strip())
        toks = []
        for t in doc:
            if t.is_punct or t.like_num: 
                continue
            w = t.lemma_.lower().strip()
            if not w or w in NLTK_STOPS or t.is_stop:
                continue
            toks.append(w)
        return " ".join(toks)

    def fit_tfidf(self, corpus: List[str]):
        norm = [self.normalize(t) for t in corpus]
        self.vec = TfidfVectorizer(max_features=75000)
        self.tfidf_matrix = self.vec.fit_transform(norm)

    def encode_tfidf(self, texts: List[str]):
        norm = [self.normalize(t) for t in texts]
        return self.vec.transform(norm)

    def _biobert(self):
        if not self._tok_bio:
            self._tok_bio = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self._mdl_bio = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(self.device).eval()
        return self._tok_bio, self._mdl_bio

    def _clinical(self):
        if not self._tok_cli:
            self._tok_cli = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self._mdl_cli = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device).eval()
        return self._tok_cli, self._mdl_cli

    @torch.inference_mode()
    def encode_text(self, texts: List[str], model_name="biobert"):
        texts = [self.normalize(t) for t in texts]
        if model_name == "biobert":
            tok, mdl = self._biobert()
        elif model_name == "clinicalbert":
            tok, mdl = self._clinical()
        else:
            raise ValueError("unknown model")
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        out = mdl(**batch)
        emb = mean_pooling(out.last_hidden_state, batch["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy()

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache, name)

    def encode_corpus(self, corpus: List[str], model_name: str):
        key = f"{model_name}_corpus.npy"
        path = self._cache_path(key)
        if os.path.exists(path): 
            return np.load(path)
        B = 24
        chunks = []
        for i in range(0, len(corpus), B):
            chunks.append(self.encode_text(corpus[i:i+B], model_name))
        mat = np.vstack(chunks)
        np.save(path, mat)
        return mat

    def build_query_from_patient(self, p: Dict[str, Any]) -> str:
        parts = []
        for k in ["Conditions","Medications","Diagnoses","Genomics","Biomarkers"]:
            v = p.get(k); 
            if v: parts.append(str(v))
        for k in ["Age","Sex"]:
            v = p.get(k); 
            if v: parts.append(str(v))
        if p.get("raw_text"): parts.append(p["raw_text"])
        return " ".join(parts).strip()

class PatientParser:
    def parse(self, text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        out = {"raw_text": t, "Age":"", "Sex":"", "Conditions":"", "Medications":""}
        
        try:
            js = json.loads(t)
            if isinstance(js, dict):
                out["Age"] = js.get("age") or js.get("Age") or ""
                out["Sex"] = js.get("sex") or js.get("Sex") or ""
                out["Conditions"] = ", ".join(js.get("conditions",[])) if isinstance(js.get("conditions",[]),list) else js.get("conditions","") or ""
                out["Medications"] = ", ".join(js.get("medications",[])) if isinstance(js.get("medications",[]),list) else js.get("medications","") or ""
                return out
        except Exception:
            pass
        # CSV single-row
        try:
            import io
            df = pd.read_csv(io.StringIO(t))
            if len(df):
                r = df.iloc[0].to_dict()
                out["Age"] = r.get("Age") or r.get("age") or ""
                out["Sex"] = r.get("Sex") or r.get("sex") or ""
                out["Conditions"] = r.get("Conditions") or r.get("conditions") or ""
                out["Medications"] = r.get("Medications") or r.get("medications") or ""
                out["raw_text"] = " ".join([str(v) for v in r.values() if pd.notna(v)])
                return out
        except Exception:
            pass
        
        m = re.search(r"\b(\d{1,3})\s*(yo|y/o|years|yrs|years old)?\b", t, flags=re.I)
        if m: out["Age"] = m.group(1)
        if re.search(r"\bfemale\b|\bf\b", t, flags=re.I): out["Sex"] = "Female"
        if re.search(r"\bmale\b|\bm\b", t, flags=re.I): out["Sex"] = "Male"
        conds=[]
        for kw in ["breast cancer","prostate cancer","leukemia","lymphoma","diabetes","asthma","hypertension",
                   "heart failure","stroke","covid","alzheimer","parkinson","metastatic","her2","egfr","brca"]:
            if re.search(r"\b"+re.escape(kw)+r"\b", t, flags=re.I):
                conds.append(kw)
        out["Conditions"] = ", ".join(sorted(set(conds)))
        return out
