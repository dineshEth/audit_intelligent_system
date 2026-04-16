from __future__ import annotations

import re
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def simple_sentence_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def simple_summary(text: str, max_sentences: int = 4) -> str:
    sentences = simple_sentence_split(text)
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences])


def keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = {tok for tok in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(tok) > 2}
    t_tokens = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(t_tokens))
    return overlap / max(len(q_tokens), 1)


def compact_reference(text: str, max_chars: int = 180) -> str:
    cleaned = normalize_whitespace(text)
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."
