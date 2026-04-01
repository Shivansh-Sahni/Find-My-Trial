from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nlp_pipeline import NLPPipeline, PatientParser
from utils import (
    LOW_SIGNAL_CONDITIONS,
    age_bucket_match,
    clean_whitespace,
    dedupe_preserve_order,
    normalize_phrase,
    shorten_text,
    split_multi_value,
)


ACTIVE_STATUSES = {
    "RECRUITING",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "AVAILABLE",
}

STATUS_SCORES = {
    "RECRUITING": 1.00,
    "ENROLLING_BY_INVITATION": 0.94,
    "AVAILABLE": 0.88,
    "NOT_YET_RECRUITING": 0.84,
    "ACTIVE_NOT_RECRUITING": 0.58,
    "UNKNOWN": 0.34,
    "COMPLETED": 0.08,
    "TERMINATED": 0.04,
    "WITHDRAWN": 0.02,
    "SUSPENDED": 0.02,
    "NO_LONGER_AVAILABLE": 0.01,
    "WITHHELD": 0.01,
}

MODE_CONFIG = {
    "conservative": {
        "word": 0.35,
        "char": 0.10,
        "semantic": 0.10,
        "condition": 0.22,
        "biomarker": 0.10,
        "therapy": 0.07,
        "stage": 0.03,
        "location": 0.03,
    },
    "balanced": {
        "word": 0.33,
        "char": 0.12,
        "semantic": 0.18,
        "condition": 0.18,
        "biomarker": 0.08,
        "therapy": 0.05,
        "stage": 0.03,
        "location": 0.03,
    },
    "exploratory": {
        "word": 0.24,
        "char": 0.10,
        "semantic": 0.28,
        "condition": 0.16,
        "biomarker": 0.07,
        "therapy": 0.07,
        "stage": 0.04,
        "location": 0.04,
    },
}

BIOMARKER_HINTS = {
    "her2": "HER2",
    "egfr": "EGFR",
    "alk": "ALK",
    "ros1": "ROS1",
    "braf": "BRAF",
    "kras": "KRAS",
    "brca": "BRCA",
    "pd l1": "PD-L1",
    "msi h": "MSI-H",
}

CONDITION_ANCHORS = (
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "sarcoma",
    "melanoma",
    "lymphoma",
    "leukemia",
    "leukaemia",
    "glioblastoma",
    "diabetes",
    "stroke",
    "asthma",
    "copd",
    "obesity",
    "schizophrenia",
    "depression",
    "hypertension",
    "covid",
    "infection",
    "failure",
    "disease",
    "syndrome",
    "hiv",
    "parkinson",
    "alzheimer",
    "als",
)


class TrialMatcher:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cache_dir: str | Path,
        use_gpu: bool = False,
        enable_semantic: bool = True,
    ):
        self.df = dataframe.fillna("")
        self.pipeline = NLPPipeline(cache_dir=cache_dir, use_gpu=use_gpu)
        self.parser = PatientParser()
        self.enable_semantic = enable_semantic
        self.records: list[dict[str, Any]] = []
        self.search_corpus: list[str] = []
        self.semantic_corpus: dict[str, np.ndarray] = {}
        self.semantic_disabled = False

        self._prepare_trials()
        self.pipeline.fit_vectorizers(self.search_corpus)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        cache_dir: str | Path,
        use_gpu: bool = False,
        enable_semantic: bool = True,
    ) -> "TrialMatcher":
        frame = pd.read_csv(csv_path)
        return cls(frame, cache_dir=cache_dir, use_gpu=use_gpu, enable_semantic=enable_semantic)

    def app_meta(self) -> dict[str, Any]:
        active_count = int(sum(record["status"] in ACTIVE_STATUSES for record in self.records))
        interventional_count = int(sum(record["is_interventional"] for record in self.records))
        return {
            "trial_count": len(self.records),
            "active_trial_count": active_count,
            "interventional_count": interventional_count,
            "semantic_enabled": self.enable_semantic and not self.semantic_disabled,
            "modes": [
                {"value": "conservative", "label": "Conservative"},
                {"value": "balanced", "label": "Balanced"},
                {"value": "exploratory", "label": "Exploratory"},
            ],
        }

    def search(
        self,
        patient_text: str,
        *,
        top_k: int = 12,
        location: str = "",
        mode: str = "balanced",
        active_only: bool = True,
        interventional_only: bool = True,
    ) -> dict[str, Any]:
        selected_mode = mode if mode in MODE_CONFIG else "balanced"
        profile = self.profile_patient(patient_text)
        query_text = self.pipeline.build_query_from_patient(profile)
        location_query = clean_whitespace(location) or (profile.get("location_hints") or [""])[0]

        word_scores = self.pipeline.word_similarity(query_text)
        char_scores = self.pipeline.char_similarity(query_text)
        semantic_scores, semantic_used = self._semantic_scores(query_text)

        diagnosis_terms = {normalize_phrase(term) for term in profile.get("diagnoses", []) if normalize_phrase(term)}
        biomarker_terms = {normalize_phrase(term) for term in profile.get("biomarkers", []) if normalize_phrase(term)}
        therapy_terms = {normalize_phrase(term) for term in profile.get("therapies", []) if normalize_phrase(term)}

        condition_scores = np.array([self._overlap_score(diagnosis_terms, record["condition_terms"]) for record in self.records])
        biomarker_scores = np.array([self._overlap_score(biomarker_terms, record["biomarker_terms"]) for record in self.records])
        therapy_scores = np.array([self._overlap_score(therapy_terms, record["intervention_terms"]) for record in self.records])
        stage_scores = np.array([self._stage_score(profile, record) for record in self.records])
        location_scores = np.array([self._location_score(location_query, record) for record in self.records])

        sex_fit = np.array([self._sex_fit(profile.get("sex"), record) for record in self.records])
        age_fit = np.array([self._age_fit(profile.get("age"), record) for record in self.records])
        status_scores = np.array([record["status_score"] for record in self.records])
        interventional_scores = np.array([1.0 if record["is_interventional"] else 0.25 for record in self.records])

        score = self._combine_scores(
            selected_mode,
            word_scores=word_scores,
            char_scores=char_scores,
            semantic_scores=semantic_scores,
            semantic_used=semantic_used,
            condition_scores=condition_scores,
            biomarker_scores=biomarker_scores,
            therapy_scores=therapy_scores,
            stage_scores=stage_scores,
            location_scores=location_scores,
            status_scores=status_scores,
            interventional_scores=interventional_scores,
            has_biomarkers=bool(biomarker_terms),
            has_therapies=bool(therapy_terms),
            has_stage=bool(profile.get("stage") or profile.get("metastatic")),
            has_location=bool(location_query),
        )

        score = score + (0.03 * status_scores)
        if not interventional_only:
            score = score + (0.02 * interventional_scores)

        mask = np.ones(len(self.records), dtype=bool)
        mask &= sex_fit > 0
        mask &= age_fit > 0
        if active_only:
            mask &= np.array([record["status"] in ACTIVE_STATUSES for record in self.records])
        if interventional_only:
            mask &= np.array([record["is_interventional"] for record in self.records])

        score = np.where(mask, score, -np.inf)
        valid_count = int(np.isfinite(score).sum())
        order = np.argsort(-score)

        components = {
            "word": word_scores,
            "char": char_scores,
            "semantic": semantic_scores,
            "condition": condition_scores,
            "biomarker": biomarker_scores,
            "therapy": therapy_scores,
            "stage": stage_scores,
            "location": location_scores,
            "status": status_scores,
        }

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(order[: max(top_k, 1)], start=1):
            if not np.isfinite(score[index]):
                continue
            record = self.records[index]
            result = self._format_result(
                profile=profile,
                record=record,
                rank=rank,
                score=float(score[index]),
                components={name: float(array[index]) for name, array in components.items()},
                location_query=location_query,
            )
            results.append(result)
            if len(results) >= top_k:
                break

        return {
            "patient": profile,
            "meta": {
                "query_text": query_text,
                "mode": selected_mode,
                "top_k": top_k,
                "active_only": active_only,
                "interventional_only": interventional_only,
                "location": location_query,
                "semantic_used": semantic_used,
                "candidate_count": valid_count,
                "trial_count": len(self.records),
                "fallback_hint": self._fallback_hint(valid_count, active_only, interventional_only),
            },
            "results": results,
        }

    def profile_patient(self, patient_text: str) -> dict[str, Any]:
        profile = self.parser.parse(patient_text)
        normalized_text = normalize_phrase(profile["raw_text"])
        tokens = normalized_text.split()
        grams = {gram for gram in self._build_patient_ngrams(tokens) if gram not in LOW_SIGNAL_CONDITIONS}

        lexicon_diagnoses = []
        if not profile.get("diagnoses"):
            lexicon_diagnoses = [
                self.condition_aliases[gram]
                for gram in grams
                if gram in self.condition_aliases and self._is_reliable_condition_term(gram)
            ]
        lexicon_therapies = [self.intervention_aliases[gram] for gram in grams if gram in self.intervention_aliases]

        profile["diagnoses"] = dedupe_preserve_order(profile.get("diagnoses", []) + lexicon_diagnoses)
        profile["therapies"] = dedupe_preserve_order(profile.get("therapies", []) + lexicon_therapies)
        profile["summary"] = self.parser._build_summary(profile)
        return profile

    def _prepare_trials(self) -> None:
        condition_counter: Counter[str] = Counter()
        intervention_counter: Counter[str] = Counter()

        for row in self.df.to_dict(orient="records"):
            conditions = dedupe_preserve_order(split_multi_value(row.get("Conditions", "")))
            conditions = [condition for condition in conditions if normalize_phrase(condition) not in LOW_SIGNAL_CONDITIONS]

            interventions = [
                self._clean_intervention(item)
                for item in split_multi_value(row.get("Interventions", ""))
            ]
            interventions = [item for item in interventions if item]

            locations = dedupe_preserve_order(split_multi_value(row.get("Locations", "")))
            title = clean_whitespace(str(row.get("Study Title", "")))
            brief_summary = clean_whitespace(str(row.get("Brief Summary", "")))
            study_type = clean_whitespace(str(row.get("Study Type", ""))).upper()
            status = clean_whitespace(str(row.get("Study Status", ""))).upper() or "UNKNOWN"
            phase = clean_whitespace(str(row.get("Phases", "")))
            sponsor = clean_whitespace(str(row.get("Sponsor", "")))

            search_text = " ".join(
                part
                for part in [
                    title,
                    brief_summary,
                    " ".join(conditions),
                    " ".join(interventions),
                    " ".join(locations),
                    phase,
                    study_type,
                ]
                if part
            )
            normalized_text = normalize_phrase(search_text)
            normalized_locations = normalize_phrase(" ".join(locations))
            condition_terms = {normalize_phrase(condition) for condition in conditions if normalize_phrase(condition)}
            intervention_terms = {normalize_phrase(item) for item in interventions if normalize_phrase(item)}
            biomarker_terms = self._extract_biomarker_terms(normalized_text)

            for condition in condition_terms:
                if len(condition.split()) <= 5 and condition not in LOW_SIGNAL_CONDITIONS:
                    condition_counter[condition] += 1
            for intervention in intervention_terms:
                if len(intervention.split()) <= 5:
                    intervention_counter[intervention] += 1

            record = {
                "nct_number": clean_whitespace(str(row.get("NCT Number", ""))),
                "title": title,
                "url": clean_whitespace(str(row.get("Study URL", ""))),
                "status": status,
                "phase": phase,
                "brief_summary": brief_summary,
                "conditions": conditions,
                "interventions": interventions,
                "locations": locations,
                "study_type": study_type,
                "sponsor": sponsor,
                "sex": clean_whitespace(str(row.get("Sex", ""))).upper() or "NA",
                "age": clean_whitespace(str(row.get("Age", ""))).upper() or "NA",
                "condition_terms": condition_terms,
                "intervention_terms": intervention_terms,
                "biomarker_terms": biomarker_terms,
                "normalized_text": normalized_text,
                "normalized_locations": normalized_locations,
                "is_interventional": "INTERVENTIONAL" in study_type,
                "status_score": STATUS_SCORES.get(status, 0.25),
                "has_advanced_signal": any(token in normalized_text for token in ("metastatic", "advanced", "recurrent", "relapsed")),
            }
            self.records.append(record)
            self.search_corpus.append(search_text)

        self.condition_aliases = {
            term: self._display_label(term)
            for term, count in condition_counter.items()
            if count >= 2 and self._is_reliable_condition_term(term)
        }
        self.intervention_aliases = {
            term: self._display_label(term)
            for term, count in intervention_counter.items()
            if count >= 1
        }

    def _semantic_scores(self, query_text: str) -> tuple[np.ndarray, bool]:
        if not self.enable_semantic or self.semantic_disabled:
            return np.zeros(len(self.records)), False

        vectors: list[np.ndarray] = []
        for model_name in ("biobert", "clinicalbert"):
            try:
                if model_name not in self.semantic_corpus:
                    self.semantic_corpus[model_name] = self.pipeline.encode_corpus(self.search_corpus, model_name)
                query_embedding = self.pipeline.encode_text([query_text], model_name=model_name)
                corpus = self.semantic_corpus[model_name]
                similarity = query_embedding @ corpus.T
                similarity = np.asarray(similarity).ravel()
                similarity = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
                vectors.append(similarity)
            except Exception:
                self.semantic_disabled = True
                return np.zeros(len(self.records)), False

        return np.mean(vectors, axis=0), True

    def _combine_scores(
        self,
        mode: str,
        *,
        word_scores: np.ndarray,
        char_scores: np.ndarray,
        semantic_scores: np.ndarray,
        semantic_used: bool,
        condition_scores: np.ndarray,
        biomarker_scores: np.ndarray,
        therapy_scores: np.ndarray,
        stage_scores: np.ndarray,
        location_scores: np.ndarray,
        status_scores: np.ndarray,
        interventional_scores: np.ndarray,
        has_biomarkers: bool,
        has_therapies: bool,
        has_stage: bool,
        has_location: bool,
    ) -> np.ndarray:
        config = MODE_CONFIG[mode]
        weighted_sum = np.zeros(len(self.records))
        total_weight = 0.0

        components = [
            ("word", word_scores, True),
            ("char", char_scores, True),
            ("semantic", semantic_scores, semantic_used),
            ("condition", condition_scores, True),
            ("biomarker", biomarker_scores, has_biomarkers),
            ("therapy", therapy_scores, has_therapies),
            ("stage", stage_scores, has_stage),
            ("location", location_scores, has_location),
        ]
        for name, values, enabled in components:
            if not enabled:
                continue
            weight = config[name]
            weighted_sum = weighted_sum + (weight * values)
            total_weight += weight

        if total_weight == 0:
            return word_scores

        blended = weighted_sum / total_weight
        blended = blended + (0.01 * status_scores)
        blended = blended + (0.01 * interventional_scores)
        return np.clip(blended, 0.0, 1.2)

    def _overlap_score(self, patient_terms: set[str], trial_terms: set[str]) -> float:
        if not patient_terms or not trial_terms:
            return 0.0
        overlap = patient_terms & trial_terms
        if not overlap:
            return 0.0
        numerator = sum(min(len(term.split()), 3) for term in overlap)
        denominator = sum(min(len(term.split()), 3) for term in patient_terms)
        return numerator / max(denominator, 1)

    def _stage_score(self, profile: dict[str, Any], record: dict[str, Any]) -> float:
        if not profile.get("stage") and not profile.get("metastatic"):
            return 0.0
        return 1.0 if record["has_advanced_signal"] else 0.0

    def _location_score(self, location_query: str, record: dict[str, Any]) -> float:
        if not location_query:
            return 0.0
        normalized_query = normalize_phrase(location_query)
        normalized_locations = record["normalized_locations"]
        if not normalized_query or not normalized_locations:
            return 0.0
        if normalized_query in normalized_locations:
            return 1.0
        query_tokens = [token for token in normalized_query.split() if len(token) > 2]
        if not query_tokens:
            return 0.0
        matches = sum(token in normalized_locations for token in query_tokens)
        return matches / len(query_tokens)

    def _sex_fit(self, patient_sex: str | None, record: dict[str, Any]) -> float:
        if not patient_sex:
            return 1.0
        allowed = record["sex"]
        if allowed in {"", "ALL", "NA"}:
            return 1.0
        if patient_sex.upper().startswith(allowed[0]):
            return 1.0
        return 0.0

    def _age_fit(self, patient_age: int | None, record: dict[str, Any]) -> float:
        return 1.0 if age_bucket_match(record["age"], patient_age) else 0.0

    def _clean_intervention(self, value: str) -> str:
        cleaned = clean_whitespace(value)
        if ":" in cleaned:
            prefix, remainder = cleaned.split(":", 1)
            if prefix.isupper():
                cleaned = remainder
        return clean_whitespace(cleaned)

    def _extract_biomarker_terms(self, normalized_text: str) -> set[str]:
        hits: set[str] = set()
        for token, label in BIOMARKER_HINTS.items():
            if token in normalized_text:
                hits.add(normalize_phrase(label))
        return hits

    def _format_result(
        self,
        *,
        profile: dict[str, Any],
        record: dict[str, Any],
        rank: int,
        score: float,
        components: dict[str, float],
        location_query: str,
    ) -> dict[str, Any]:
        diagnosis_terms = {normalize_phrase(term) for term in profile.get("diagnoses", []) if normalize_phrase(term)}
        biomarker_terms = {normalize_phrase(term) for term in profile.get("biomarkers", []) if normalize_phrase(term)}
        therapy_terms = {normalize_phrase(term) for term in profile.get("therapies", []) if normalize_phrase(term)}

        matched_conditions = self._display_matches(diagnosis_terms & record["condition_terms"])
        matched_biomarkers = self._display_matches(biomarker_terms & record["biomarker_terms"])
        matched_therapies = self._display_matches(therapy_terms & record["intervention_terms"])

        reasons: list[str] = []
        cautions: list[str] = []

        if matched_conditions:
            reasons.append("Condition overlap: " + ", ".join(matched_conditions[:4]))
        if matched_biomarkers:
            reasons.append("Biomarker signal: " + ", ".join(matched_biomarkers[:4]))
        if matched_therapies:
            reasons.append("Therapy context overlap: " + ", ".join(matched_therapies[:4]))
        if record["status"] in ACTIVE_STATUSES:
            reasons.append("Operationally relevant status: " + self._status_label(record["status"]))
        if location_query and components["location"] > 0:
            reasons.append("Location match against " + location_query)
        if record["is_interventional"]:
            reasons.append("Interventional study design")
        if profile.get("stage") and record["has_advanced_signal"]:
            reasons.append("Advanced/metastatic disease language present in trial text")

        if record["status"] not in ACTIVE_STATUSES:
            cautions.append("Status is " + self._status_label(record["status"]))
        if location_query and components["location"] == 0:
            cautions.append("No clear location hit for " + location_query)
        if not matched_biomarkers and profile.get("biomarkers"):
            cautions.append("Patient biomarkers were not explicit in the indexed trial text")
        if not record["phase"]:
            cautions.append("Phase is not specified in this sample row")

        score_pct = round(float(np.clip(score, 0.0, 1.0) * 100), 1)
        fit_label = self._fit_label(score_pct)

        breakdown = {
            "lexical": round(((components["word"] * 0.65) + (components["char"] * 0.35)) * 100, 1),
            "semantic": round(components["semantic"] * 100, 1),
            "condition": round(components["condition"] * 100, 1),
            "biomarker": round(components["biomarker"] * 100, 1),
            "therapy": round(components["therapy"] * 100, 1),
            "stage": round(components["stage"] * 100, 1),
            "location": round(components["location"] * 100, 1),
            "status": round(components["status"] * 100, 1),
        }

        return {
            "rank": rank,
            "score": score_pct,
            "fit_label": fit_label,
            "nct_number": record["nct_number"],
            "title": record["title"],
            "url": record["url"],
            "status": self._status_label(record["status"]),
            "phase": record["phase"] or "Not specified",
            "study_type": record["study_type"].replace("_", " ").title(),
            "sponsor": record["sponsor"],
            "brief_summary": shorten_text(record["brief_summary"], 420),
            "conditions": record["conditions"][:6],
            "interventions": record["interventions"][:6],
            "locations": record["locations"][:3],
            "reasons": dedupe_preserve_order(reasons)[:4],
            "cautions": dedupe_preserve_order(cautions)[:3],
            "matched_conditions": matched_conditions,
            "matched_biomarkers": matched_biomarkers,
            "matched_therapies": matched_therapies,
            "breakdown": breakdown,
            "eligibility": {
                "sex": record["sex"] if record["sex"] != "NA" else "Not specified",
                "age": record["age"] if record["age"] != "NA" else "Not specified",
            },
        }

    def _display_matches(self, terms: set[str]) -> list[str]:
        return [self._display_label(term) for term in sorted(terms, key=lambda item: (-len(item.split()), item))]

    def _is_reliable_condition_term(self, term: str) -> bool:
        if term in LOW_SIGNAL_CONDITIONS:
            return False
        return any(anchor in term for anchor in CONDITION_ANCHORS)

    def _display_label(self, term: str) -> str:
        words = []
        for piece in term.split():
            if piece.upper() in {"EGFR", "HER2", "ALK", "ROS1", "BRCA", "BRAF", "KRAS"}:
                words.append(piece.upper())
            elif piece == "pd":
                words.append("PD")
            elif piece == "l1":
                words.append("L1")
            elif piece == "msi":
                words.append("MSI")
            elif piece == "h":
                words.append("H")
            else:
                words.append(piece.capitalize())
        label = " ".join(words)
        label = label.replace("Pd L1", "PD-L1").replace("Msi H", "MSI-H")
        return label

    def _status_label(self, status: str) -> str:
        return status.replace("_", " ").title()

    def _fit_label(self, score: float) -> str:
        if score >= 78:
            return "High Conviction"
        if score >= 60:
            return "Promising"
        return "Exploratory"

    def _fallback_hint(self, valid_count: int, active_only: bool, interventional_only: bool) -> str | None:
        if valid_count > 0:
            return None
        if active_only and interventional_only:
            return "No active interventional trials cleared the demographic filters. Try Exploratory mode or relax one filter."
        if active_only:
            return "No active/upcoming trials cleared the current filters. Try relaxing the status filter."
        return "No trials cleared the current filters. Try broader chart text or Exploratory mode."

    def _build_patient_ngrams(self, tokens: list[str]) -> set[str]:
        grams: set[str] = set()
        for size in range(1, min(5, len(tokens)) + 1):
            for index in range(0, len(tokens) - size + 1):
                grams.add(" ".join(tokens[index : index + size]))
        return grams
