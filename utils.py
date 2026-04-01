from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CLINICAL_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "patient",
    "patients",
    "study",
    "trial",
    "clinical",
    "history",
    "present",
    "disease",
    "diseases",
    "status",
    "treatment",
    "treatments",
    "therapy",
    "therapies",
    "mg",
    "ml",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "old",
}

LOW_SIGNAL_CONDITIONS = {
    "cancer",
    "advanced cancer",
    "solid tumors",
    "tumor",
    "healthy",
    "pain",
    "disease",
    "condition",
}

RTF_HEX_RE = re.compile(r"\\'([0-9a-fA-F]{2})")
RTF_UNICODE_RE = re.compile(r"\\u(-?\d+)\??")
RTF_CONTROL_RE = re.compile(r"\\[a-zA-Z]+-?\d* ?")


def clean_whitespace(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_rtf(text: str) -> str:
    text = text or ""
    if "\\rtf" not in text[:64].lower():
        return clean_whitespace(text)

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\\par", "\n").replace("\\line", "\n").replace("\\tab", "\t")
    cleaned = RTF_HEX_RE.sub(lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), cleaned)
    cleaned = RTF_UNICODE_RE.sub(_decode_rtf_unicode, cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = RTF_CONTROL_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\\", "\n")
    return clean_whitespace(cleaned)


def _decode_rtf_unicode(match: re.Match[str]) -> str:
    codepoint = int(match.group(1))
    if codepoint < 0:
        return " "
    try:
        return chr(codepoint)
    except ValueError:
        return " "


def normalize_phrase(text: str) -> str:
    text = strip_rtf(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"([a-z0-9])\+", r"\1 positive", text)
    text = re.sub(r"\btriple[- ]negative\b", "triple negative", text)
    text = re.sub(r"\bher2[- ]positive\b", "her2 positive", text)
    text = re.sub(r"\begfr[- ]mutated\b", "egfr mutated", text)
    text = re.sub(r"[/_]", " ", text)
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_multi_value(value: object, *, split_commas: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(item) for item in value]
    else:
        parts = [str(value)]

    items: list[str] = []
    splitter = r"[|\n;]" if not split_commas else r"[|\n;,]"
    for part in parts:
        for fragment in re.split(splitter, part):
            cleaned = clean_whitespace(fragment)
            if cleaned:
                items.append(cleaned)
    return items


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def build_ngrams(tokens: Sequence[str], max_n: int = 5) -> set[str]:
    grams: set[str] = set()
    upper = min(max_n, len(tokens))
    for size in range(1, upper + 1):
        for index in range(0, len(tokens) - size + 1):
            grams.add(" ".join(tokens[index : index + size]))
    return grams


def shorten_text(text: str, limit: int = 320) -> str:
    text = clean_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "..."


def coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def age_to_bucket(age: int | None) -> str | None:
    if age is None:
        return None
    if age < 18:
        return "CHILD"
    if age < 65:
        return "ADULT"
    return "OLDER_ADULT"


def age_bucket_match(age_label: str, age: int | None) -> bool:
    if not age_label or age is None:
        return True
    bucket = age_to_bucket(age)
    normalized = {part.strip().upper() for part in age_label.split(",") if part.strip()}
    if not normalized or "NA" in normalized:
        return True
    return bucket in normalized


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / counts


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T
