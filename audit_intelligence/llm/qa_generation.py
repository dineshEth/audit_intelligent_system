from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from ..utils.text import normalize_whitespace, simple_summary
from .prompt_templates import QA_GENERATION_PROMPT


def _parse_json_list(text: str) -> List[Dict[str, str]]:
    text = text.strip()
    if not text:
        return []
    candidates = [text]
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    elif "```" in text:
        candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, list):
                return [
                    {"question": str(item.get("question", "")).strip(), "answer": str(item.get("answer", "")).strip()}
                    for item in payload
                    if isinstance(item, dict)
                ]
        except Exception:
            continue
    return []


class QAGenerator:
    def __init__(self, llm) -> None:
        self.llm = llm

    def from_bank_statement(self, dataframe: pd.DataFrame, analysis: Dict[str, Any], max_pairs: int = 12) -> List[Dict[str, str]]:
        df = dataframe.copy()
        if df.empty:
            return []

        qas: List[Dict[str, str]] = []
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        period_start = df["DATE"].min().date().isoformat()
        period_end = df["DATE"].max().date().isoformat()
        opening_balance = round(float(df["BALANCE"].iloc[0]), 2)
        closing_balance = round(float(df["BALANCE"].iloc[-1]), 2)
        total_debit = round(float(df["DEBIT"].sum()), 2)
        total_credit = round(float(df["CREDIT"].sum()), 2)

        qas.extend(
            [
                {"question": "What period does the statement cover?", "answer": f"The statement covers {period_start} to {period_end}."},
                {"question": "What is the opening balance?", "answer": f"The opening balance is {opening_balance}."},
                {"question": "What is the closing balance?", "answer": f"The closing balance is {closing_balance}."},
                {"question": "What is the total debit?", "answer": f"The total debit is {total_debit}."},
                {"question": "What is the total credit?", "answer": f"The total credit is {total_credit}."},
                {"question": "How many transactions are present?", "answer": f"There are {len(df)} transactions in the statement."},
            ]
        )

        if "CATEGORY" in df.columns:
            category_totals = (
                df.assign(AMOUNT=df["DEBIT"].fillna(0) + df["CREDIT"].fillna(0))
                .groupby("CATEGORY")["AMOUNT"]
                .sum()
                .sort_values(ascending=False)
            )
            if not category_totals.empty:
                top_category = str(category_totals.index[0])
                qas.append(
                    {
                        "question": "Which transaction category has the highest volume?",
                        "answer": f"The highest-volume category is {top_category}.",
                    }
                )

        largest_debit = df.sort_values("DEBIT", ascending=False).iloc[0]
        largest_credit = df.sort_values("CREDIT", ascending=False).iloc[0]
        qas.extend(
            [
                {
                    "question": "What is the largest debit transaction?",
                    "answer": f"The largest debit is {float(largest_debit['DEBIT']):.2f} for {largest_debit['DESCRIPTION']}.",
                },
                {
                    "question": "What is the largest credit transaction?",
                    "answer": f"The largest credit is {float(largest_credit['CREDIT']):.2f} for {largest_credit['DESCRIPTION']}.",
                },
            ]
        )

        anomalies = analysis.get("anomalies", []) if analysis else []
        if anomalies:
            qas.append(
                {
                    "question": "Were any anomalous transactions detected?",
                    "answer": f"Yes. {len(anomalies)} unusually large transactions were flagged.",
                }
            )

        return qas[:max_pairs]

    def from_text(self, text: str, max_pairs: int = 10) -> List[Dict[str, str]]:
        context = text[:4000]
        if self.llm.available:
            prompt = QA_GENERATION_PROMPT.format(context=context)
            generated = self.llm.generate(prompt, max_new_tokens=600, temperature=0.0)
            parsed = _parse_json_list(generated)
            if parsed:
                return parsed[:max_pairs]

        summary = simple_summary(context, max_sentences=4)
        return [
            {"question": "What is the main topic of the document?", "answer": summary or "The document contains financial information."},
            {"question": "What kind of content appears in the document?", "answer": "The document contains text that can be indexed for audit retrieval and summarization."},
            {"question": "How should this document be used in the audit system?", "answer": "It should be ingested, chunked, stored, and used for retrieval-augmented responses."},
        ][:max_pairs]

    def build_pairs(self, text: str, dataframe: pd.DataFrame | None = None, analysis: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
        if dataframe is not None and not dataframe.empty:
            return self.from_bank_statement(dataframe, analysis or {})
        return self.from_text(text)
