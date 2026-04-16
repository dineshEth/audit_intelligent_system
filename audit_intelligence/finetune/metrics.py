from __future__ import annotations

import re
from typing import Iterable


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def exact_match(prediction: str, target: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(target) else 0.0

def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(target).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    gold_counts = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    common = 0
    for token, count in pred_counts.items():
        common += min(count, gold_counts.get(token, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
