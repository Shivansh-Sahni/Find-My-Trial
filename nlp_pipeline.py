from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from utils import (
    CLINICAL_STOPWORDS,
    clean_whitespace,
    coerce_int,
    dedupe_preserve_order,
    mean_pooling,
    normalize_phrase,
    split_multi_value,
    strip_rtf,
)


SECTION_HEADINGS = {
    "chief complaint",
    "history of present illness",
    "present illness",
    "past medical history",
    "medical history",
    "surgical history",
    "allergies",
    "medications",
    "family history",
    "social history",
    "review of systems",
    "physical examination",
    "laboratory results",
    "labs",
    "imaging",
    "performance status",
    "assessment",
    "plan",
    "diagnoses",
}

DISEASE_PATTERNS = {
    "Breast Cancer": [
        r"\bbreast cancer\b",
        r"\binvasive ductal carcinoma\b",
    ],
    "Lung Cancer": [r"\blung cancer\b", r"\bnon[- ]small cell lung cancer\b", r"\bnsclc\b"],
    "Small Cell Lung Cancer": [r"\bsmall cell lung cancer\b", r"\bsclc\b"],
    "Colorectal Cancer": [r"\bcolorectal cancer\b", r"\bcolon cancer\b", r"\brectal cancer\b"],
    "Prostate Cancer": [r"\bprostate cancer\b"],
    "Pancreatic Cancer": [r"\bpancreatic cancer\b"],
    "Ovarian Cancer": [r"\bovarian cancer\b"],
    "Leukemia": [r"\bleukemia\b"],
    "Lymphoma": [r"\blymphoma\b"],
    "Melanoma": [r"\bmelanoma\b"],
    "Glioblastoma": [r"\bglioblastoma\b", r"\bgbm\b"],
    "Sarcoma": [r"\bsarcoma\b"],
    "Stroke": [r"\bstroke\b", r"\bcerebrovascular accident\b"],
    "Heart Failure": [r"\bheart failure\b"],
    "Hypertension": [r"\bhypertension\b", r"\bhigh blood pressure\b"],
    "Type 2 Diabetes": [r"\btype 2 diabetes\b", r"\bdiabetes mellitus type 2\b"],
    "Diabetes": [r"\bdiabetes\b"],
    "Asthma": [r"\basthma\b"],
    "COPD": [r"\bcopd\b", r"\bchronic obstructive pulmonary disease\b"],
    "Schizophrenia": [r"\bschizophrenia\b"],
    "Depression": [r"\bdepression\b"],
    "COVID-19": [r"\bcovid(?:-19)?\b", r"\bsars cov 2\b"],
    "ALS": [r"\bamyotrophic lateral sclerosis\b", r"\bals\b"],
}

COMORBIDITY_PATTERNS = {
    "Hypertension": [r"\bhypertension\b", r"\bhigh blood pressure\b"],
    "Type 2 Diabetes": [r"\btype 2 diabetes\b", r"\bdiabetes mellitus\b", r"\bdiabetes\b"],
    "Hyperlipidemia": [r"\bhyperlipidemia\b", r"\bhigh cholesterol\b"],
    "Heart Failure": [r"\bheart failure\b"],
    "Asthma": [r"\basthma\b"],
    "COPD": [r"\bcopd\b", r"\bchronic obstructive pulmonary disease\b"],
    "Depression": [r"\bdepression\b"],
    "Stroke": [r"\bstroke\b"],
}

BIOMARKER_PATTERNS = {
    "HER2": [r"\bher2(?: positive)?\b", r"\berbb2\b"],
    "EGFR": [r"\begfr\b"],
    "ALK": [r"\balk\b"],
    "ROS1": [r"\bros1\b"],
    "BRAF": [r"\bbraf\b"],
    "KRAS": [r"\bkras\b"],
    "BRCA": [r"\bbrca(?:1|2)?\b"],
    "PD-L1": [r"\bpd[\s-]?l1\b"],
    "MSI-H": [r"\bmsi[- ]h\b", r"\bmicrosatellite instability high\b"],
    "HR+": [r"\bhr positive\b", r"\bhormone receptor positive\b"],
}

THERAPY_PATTERNS = {
    "Trastuzumab": [r"\btrastuzumab\b", r"\bherceptin\b"],
    "Pertuzumab": [r"\bpertuzumab\b"],
    "Paclitaxel": [r"\bpaclitaxel\b"],
    "Carboplatin": [r"\bcarboplatin\b"],
    "Cisplatin": [r"\bcisplatin\b"],
    "Doxorubicin": [r"\bdoxorubicin\b"],
    "Cyclophosphamide": [r"\bcyclophosphamide\b"],
    "Pembrolizumab": [r"\bpembrolizumab\b"],
    "Nivolumab": [r"\bnivolumab\b"],
    "Osimertinib": [r"\bosimertinib\b"],
    "Tucatinib": [r"\btucatinib\b"],
    "Chemotherapy": [r"\bchemotherapy\b"],
    "Immunotherapy": [r"\bimmunotherapy\b"],
    "Radiation Therapy": [r"\bradiation therapy\b"],
    "Surgery": [r"\blumpectomy\b", r"\bmastectomy\b", r"\bsurgery\b"],
}

METASTATIC_SITE_PATTERNS = {
    "Bone": [r"\bbone metast", r"\bosseous metast"],
    "Liver": [r"\bliver metast", r"\bhepatic metast"],
    "Lung": [r"\blung metast", r"\bpulmonary metast"],
    "Brain": [r"\bbrain metast"],
}

MODEL_NAMES = {
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


class NLPPipeline:
    def __init__(self, cache_dir: str | Path = "./.cache", use_gpu: bool = False):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.word_matrix = None
        self.char_matrix = None
        self._models: dict[str, tuple[Any, Any]] = {}
        self._disabled_models: set[str] = set()

    def normalize(self, text: str) -> str:
        tokens = normalize_phrase(strip_rtf(text)).split()
        filtered = []
        for token in tokens:
            if token in CLINICAL_STOPWORDS:
                continue
            if token.isdigit():
                continue
            if len(token) < 2 and not any(char.isdigit() for char in token):
                continue
            filtered.append(token)
        return " ".join(filtered)

    def fit_vectorizers(self, corpus: list[str]) -> None:
        normalized = [self.normalize(text) for text in corpus]
        self.word_vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=40000,
            min_df=2,
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(normalized)
        self.char_matrix = self.char_vectorizer.fit_transform(normalized)

    def word_similarity(self, query_text: str) -> np.ndarray:
        query = self.word_vectorizer.transform([self.normalize(query_text)])
        return np.asarray((query @ self.word_matrix.T).todense()).ravel()

    def char_similarity(self, query_text: str) -> np.ndarray:
        query = self.char_vectorizer.transform([self.normalize(query_text)])
        return np.asarray((query @ self.char_matrix.T).todense()).ravel()

    def _get_model(self, model_name: str):
        if model_name in self._disabled_models:
            raise RuntimeError(f"{model_name} is disabled")
        if model_name not in self._models:
            huggingface_name = MODEL_NAMES[model_name]
            try:
                tokenizer = AutoTokenizer.from_pretrained(huggingface_name, local_files_only=True)
                model = AutoModel.from_pretrained(huggingface_name, local_files_only=True)
            except Exception as exc:
                self._disabled_models.add(model_name)
                raise RuntimeError(str(exc)) from exc
            self._models[model_name] = (tokenizer, model.to(self.device).eval())
        return self._models[model_name]

    @torch.inference_mode()
    def encode_text(self, texts: list[str], model_name: str = "biobert") -> np.ndarray:
        tokenizer, model = self._get_model(model_name)
        cleaned = [clean_whitespace(text) for text in texts]
        batch = tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        output = model(**batch)
        embedding = mean_pooling(output.last_hidden_state, batch["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.detach().cpu().numpy()

    def encode_corpus(self, corpus: list[str], model_name: str) -> np.ndarray:
        cache_path = self.cache / f"{model_name}_corpus.npy"
        if cache_path.exists():
            return np.load(cache_path)
        encoded: list[np.ndarray] = []
        batch_size = 24
        for start in range(0, len(corpus), batch_size):
            encoded.append(self.encode_text(corpus[start : start + batch_size], model_name=model_name))
        matrix = np.vstack(encoded)
        np.save(cache_path, matrix)
        return matrix

    def build_query_from_patient(self, profile: dict[str, Any]) -> str:
        parts: list[str] = []
        if profile.get("summary"):
            parts.append(profile["summary"])
        for field in ("diagnoses", "biomarkers", "therapies", "comorbidities", "metastatic_sites"):
            parts.extend(profile.get(field, []))
        if profile.get("stage"):
            parts.append(profile["stage"])
        if profile.get("performance_status"):
            parts.append(profile["performance_status"])
        if profile.get("age"):
            parts.append(f"{profile['age']} year old")
        if profile.get("sex"):
            parts.append(profile["sex"])
        for sentence in profile.get("salient_sentences", [])[:4]:
            parts.append(sentence)
        return " ".join(dedupe_preserve_order(clean_whitespace(part) for part in parts if part))


class PatientParser:
    def parse(self, text: str) -> dict[str, Any]:
        structured_payload = self._parse_structured_payload(text)
        clean_text = structured_payload["raw_text"] if structured_payload else strip_rtf(text)
        clean_text = clean_whitespace(clean_text)
        sections, facts = self._extract_sections_and_facts(clean_text)

        diagnoses = structured_payload.get("conditions", []) if structured_payload else []
        therapies = structured_payload.get("medications", []) if structured_payload else []

        diagnosis_focus = self._compose_section_slice(
            sections,
            ["chief complaint", "history of present illness", "assessment", "plan", "diagnoses"],
        )
        if not diagnosis_focus:
            diagnosis_focus = clean_text

        diagnoses.extend(self._extract_pattern_matches(diagnosis_focus, DISEASE_PATTERNS))
        diagnoses = self._normalize_labels(dedupe_preserve_order(diagnoses))

        biomarkers = self._normalize_labels(self._extract_pattern_matches(clean_text, BIOMARKER_PATTERNS))
        therapies.extend(self._extract_pattern_matches(clean_text, THERAPY_PATTERNS))
        therapies.extend(self._extract_medications_from_section(sections.get("medications", "")))
        therapies = self._normalize_labels(dedupe_preserve_order(therapies))

        comorbidities = self._normalize_labels(
            dedupe_preserve_order(self._extract_pattern_matches(self._compose_section_slice(sections, ["past medical history", "medical history"]), COMORBIDITY_PATTERNS))
        )

        age = structured_payload.get("age") if structured_payload else None
        if age is None:
            age = self._extract_age(clean_text, facts)

        sex = structured_payload.get("sex") if structured_payload else None
        if not sex:
            sex = self._extract_sex(clean_text, facts)

        stage = self._extract_stage(clean_text)
        performance_status = self._extract_performance_status(clean_text, sections, facts)
        metastatic = bool(re.search(r"\bmetastatic\b|\bmetastases\b|\bmetastasis\b|\bstage iv\b", normalize_phrase(clean_text)))
        metastatic_sites = self._normalize_labels(self._extract_pattern_matches(clean_text, METASTATIC_SITE_PATTERNS))
        location_hints = self._extract_location_hints(clean_text)
        salient_sentences = self._extract_salient_sentences(clean_text)

        profile = {
            "raw_text": clean_text,
            "age": age,
            "sex": sex,
            "diagnoses": diagnoses,
            "biomarkers": biomarkers,
            "therapies": therapies,
            "comorbidities": comorbidities,
            "stage": stage,
            "performance_status": performance_status,
            "metastatic": metastatic,
            "metastatic_sites": metastatic_sites,
            "location_hints": location_hints,
            "salient_sentences": salient_sentences,
            "sections": sections,
        }
        profile["summary"] = self._build_summary(profile)
        return profile

    def _parse_structured_payload(self, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}

        try:
            loaded = json.loads(raw)
        except Exception:
            loaded = None

        if isinstance(loaded, dict):
            return {
                "age": coerce_int(loaded.get("age") or loaded.get("Age")),
                "sex": self._canonical_sex(loaded.get("sex") or loaded.get("Sex")),
                "conditions": self._coerce_string_list(loaded.get("conditions") or loaded.get("Conditions")),
                "medications": self._coerce_string_list(loaded.get("medications") or loaded.get("Medications")),
                "raw_text": clean_whitespace(" ".join(self._flatten_values(loaded))),
            }

        try:
            dataframe = pd.read_csv(io.StringIO(raw))
        except Exception:
            dataframe = None

        if dataframe is not None and not dataframe.empty:
            row = dataframe.iloc[0].to_dict()
            return {
                "age": coerce_int(row.get("Age") or row.get("age")),
                "sex": self._canonical_sex(row.get("Sex") or row.get("sex")),
                "conditions": self._coerce_string_list(row.get("Conditions") or row.get("conditions")),
                "medications": self._coerce_string_list(row.get("Medications") or row.get("medications")),
                "raw_text": clean_whitespace(" ".join(str(value) for value in row.values() if pd.notna(value))),
            }

        return {}

    def _flatten_values(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            flattened: list[str] = []
            for nested in value.values():
                flattened.extend(self._flatten_values(nested))
            return flattened
        if isinstance(value, list):
            flattened: list[str] = []
            for nested in value:
                flattened.extend(self._flatten_values(nested))
            return flattened
        return [str(value)] if value is not None else []

    def _extract_sections_and_facts(self, text: str) -> tuple[dict[str, str], dict[str, str]]:
        sections: dict[str, list[str]] = {}
        facts: dict[str, str] = {}
        current_section: str | None = None

        for raw_line in text.splitlines():
            line = clean_whitespace(raw_line)
            if not line:
                continue

            key, value = self._split_fact_line(line)
            if key and key in SECTION_HEADINGS and not value:
                current_section = key
                sections.setdefault(current_section, [])
                continue
            if key and key in SECTION_HEADINGS and value:
                current_section = key
                sections.setdefault(current_section, []).append(value)
                continue
            if key and value and len(value) < 120:
                facts[key] = value
            if current_section:
                sections.setdefault(current_section, []).append(line)

        joined_sections = {name: clean_whitespace("\n".join(lines)) for name, lines in sections.items()}
        return joined_sections, facts

    def _split_fact_line(self, line: str) -> tuple[str | None, str]:
        match = re.match(r"^([A-Za-z][A-Za-z /&()\-]{1,40}):\s*(.*)$", line)
        if not match:
            return None, ""
        heading = match.group(1).strip().lower()
        value = match.group(2).strip()
        return heading, value

    def _compose_section_slice(self, sections: dict[str, str], names: list[str]) -> str:
        return "\n".join(sections.get(name, "") for name in names if sections.get(name))

    def _extract_pattern_matches(self, text: str, catalog: dict[str, list[str]]) -> list[str]:
        normalized = normalize_phrase(text)
        matches: list[str] = []
        for label, patterns in catalog.items():
            for pattern in patterns:
                if re.search(pattern, normalized, flags=re.IGNORECASE):
                    matches.append(label)
                    break
        return matches

    def _extract_medications_from_section(self, section_text: str) -> list[str]:
        if not section_text:
            return []
        extracted: list[str] = []
        for piece in split_multi_value(section_text, split_commas=True):
            cleaned = re.sub(r"\(.*?\)", "", piece)
            cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b.*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(?:bid|tid|qid|daily|nightly|prn)\b.*", "", cleaned, flags=re.IGNORECASE)
            cleaned = clean_whitespace(cleaned.strip("- "))
            if 2 <= len(cleaned) <= 48:
                extracted.append(cleaned)
        return extracted

    def _extract_age(self, text: str, facts: dict[str, str]) -> int | None:
        for key in ("age",):
            if key in facts:
                value = coerce_int(facts[key])
                if value is not None:
                    return value
        match = re.search(r"\b(\d{1,3})\s*(?:year old|years old|yo|y/o|yrs old|yrs)\b", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_sex(self, text: str, facts: dict[str, str]) -> str | None:
        if "sex" in facts:
            return self._canonical_sex(facts["sex"])
        normalized = normalize_phrase(text)
        if re.search(r"\bfemale\b|\bwoman\b", normalized):
            return "Female"
        if re.search(r"\bmale\b|\bman\b", normalized):
            return "Male"
        return None

    def _extract_stage(self, text: str) -> str | None:
        normalized = normalize_phrase(text)
        if re.search(r"\bstage iv\b", normalized):
            return "Stage IV"
        match = re.search(r"\bstage ([i,vx]+)\b", normalized)
        if match:
            return f"Stage {match.group(1).upper()}"
        if re.search(r"\bmetastatic\b|\bmetastases\b", normalized):
            return "Metastatic"
        return None

    def _extract_performance_status(self, text: str, sections: dict[str, str], facts: dict[str, str]) -> str | None:
        for candidate in [facts.get("performance status", ""), sections.get("performance status", ""), text]:
            match = re.search(r"\becog\s*([0-5])\b", candidate, flags=re.IGNORECASE)
            if match:
                return f"ECOG {match.group(1)}"
            match = re.search(r"\bkarnofsky\s*(?:score)?\s*(\d{2,3})\b", candidate, flags=re.IGNORECASE)
            if match:
                return f"Karnofsky {match.group(1)}"
        return None

    def _extract_location_hints(self, text: str) -> list[str]:
        hints: list[str] = []
        for match in re.finditer(r"\b(?:lives in|located in|from)\s+([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)?)", text):
            hints.append(match.group(1))
        return dedupe_preserve_order(hints)

    def _extract_salient_sentences(self, text: str) -> list[str]:
        normalized = clean_whitespace(text)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        preferred: list[str] = []
        for sentence in sentences:
            compact = clean_whitespace(sentence)
            if len(compact) < 25:
                continue
            if any(
                token in normalize_phrase(compact)
                for token in ("cancer", "metastatic", "ecog", "trastuzumab", "pertuzumab", "her2", "egfr", "trial")
            ):
                preferred.append(compact)
        return preferred[:5]

    def _coerce_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [clean_whitespace(str(item)) for item in value if clean_whitespace(str(item))]
        return split_multi_value(str(value), split_commas=True)

    def _normalize_labels(self, items: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            label = clean_whitespace(item)
            if not label:
                continue
            key = normalize_phrase(label)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(label)
        return normalized

    def _canonical_sex(self, value: Any) -> str | None:
        normalized = normalize_phrase(str(value))
        if normalized in {"female", "f"}:
            return "Female"
        if normalized in {"male", "m"}:
            return "Male"
        return None

    def _build_summary(self, profile: dict[str, Any]) -> str:
        fragments: list[str] = []
        demographic = []
        if profile.get("age"):
            demographic.append(f"{profile['age']}-year-old")
        if profile.get("sex"):
            demographic.append(profile["sex"].lower())
        if demographic:
            fragments.append(" ".join(demographic))
        if profile.get("stage"):
            fragments.append(profile["stage"])
        if profile.get("diagnoses"):
            diagnosis = ", ".join(profile["diagnoses"][:3])
            fragments.append(diagnosis)
        if profile.get("biomarkers"):
            fragments.append("biomarkers " + ", ".join(profile["biomarkers"][:4]))
        if profile.get("therapies"):
            fragments.append("prior/current therapies " + ", ".join(profile["therapies"][:4]))
        if profile.get("performance_status"):
            fragments.append(profile["performance_status"])
        if profile.get("comorbidities"):
            fragments.append("comorbidities " + ", ".join(profile["comorbidities"][:3]))
        return "; ".join(fragments)
