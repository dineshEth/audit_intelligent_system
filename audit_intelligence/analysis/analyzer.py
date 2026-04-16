from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..utils.text import compact_reference, simple_summary


class FinancialAnalyzer:
    def analyze_transactions(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        if dataframe.empty:
            return {
                "transaction_count": 0,
                "total_debit": 0.0,
                "total_credit": 0.0,
                "opening_balance": 0.0,
                "closing_balance": 0.0,
                "category_totals": {},
                "anomalies": [],
            }

        df = dataframe.copy()
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df["DEBIT"] = pd.to_numeric(df["DEBIT"], errors="coerce").fillna(0.0)
        df["CREDIT"] = pd.to_numeric(df["CREDIT"], errors="coerce").fillna(0.0)
        df["BALANCE"] = pd.to_numeric(df["BALANCE"], errors="coerce").fillna(0.0)
        amount = df["DEBIT"].where(df["DEBIT"] > 0, df["CREDIT"])
        threshold = float(amount.mean() + (2 * amount.std())) if len(df) > 1 else float(amount.max())

        anomalies_df = df[amount >= threshold].head(10)
        anomalies = [
            {
                "date": row["DATE"].date().isoformat() if not pd.isna(row["DATE"]) else "",
                "description": row["DESCRIPTION"],
                "debit": float(row["DEBIT"]),
                "credit": float(row["CREDIT"]),
                "balance": float(row["BALANCE"]),
            }
            for _, row in anomalies_df.iterrows()
        ]

        category_totals = {}
        if "CATEGORY" in df.columns:
            category_totals = (
                df.assign(AMOUNT=df["DEBIT"].where(df["DEBIT"] > 0, df["CREDIT"]))
                .groupby("CATEGORY")["AMOUNT"]
                .sum()
                .round(2)
                .to_dict()
            )

        merchant_totals = (
            df.assign(AMOUNT=df["DEBIT"].where(df["DEBIT"] > 0, df["CREDIT"]))
            .groupby("DESCRIPTION")["AMOUNT"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )

        return {
            "transaction_count": int(len(df)),
            "period_start": df["DATE"].min().date().isoformat(),
            "period_end": df["DATE"].max().date().isoformat(),
            "total_debit": round(float(df["DEBIT"].sum()), 2),
            "total_credit": round(float(df["CREDIT"].sum()), 2),
            "net_cash_flow": round(float(df["CREDIT"].sum() - df["DEBIT"].sum()), 2),
            "opening_balance": round(float(df["BALANCE"].iloc[0]), 2),
            "closing_balance": round(float(df["BALANCE"].iloc[-1]), 2),
            "category_totals": category_totals,
            "top_merchants": merchant_totals,
            "anomalies": anomalies,
        }

    def summarize_generic_document(self, text: str, retrieved_chunks: List[dict] | None = None) -> Dict[str, Any]:
        retrieved_chunks = retrieved_chunks or []
        references = [{"snippet": compact_reference(item.get("text", "")), "score": item.get("score", 0.0)} for item in retrieved_chunks[:5]]
        return {
            "summary": simple_summary(text, max_sentences=5),
            "references": references,
        }
