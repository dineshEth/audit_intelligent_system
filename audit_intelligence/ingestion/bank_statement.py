from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .loaders import load_file


COLUMN_ALIASES = {
    "DATE": ["date", "txn date", "transaction date", "value date", "posted date"],
    "DESCRIPTION": ["description", "details", "narration", "remarks", "merchant", "transaction details"],
    "DEBIT": ["debit", "withdrawal", "dr", "debit amount"],
    "CREDIT": ["credit", "deposit", "cr", "credit amount"],
    "BALANCE": ["balance", "running balance", "closing balance", "available balance"],
    "AMOUNT": ["amount", "txn amount"],
}


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()


def _find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized = {_normalize_column(col): col for col in columns}
    for alias in aliases:
        key = _normalize_column(alias)
        if key in normalized:
            return normalized[key]
    return None


def _clean_amount(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(",", "").replace("₹", "").strip()
    if cleaned in {"", "-", "nan", "None"}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        return float(match.group(0)) if match else 0.0


def normalize_bank_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = list(dataframe.columns)
    mapping: Dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        found = _find_column(columns, aliases)
        if found:
            mapping[found] = target

    df = dataframe.rename(columns=mapping).copy()

    if "DATE" not in df.columns:
        raise ValueError("Unable to locate a DATE column in the bank statement.")
    if "DESCRIPTION" not in df.columns:
        raise ValueError("Unable to locate a DESCRIPTION column in the bank statement.")

    if "DEBIT" not in df.columns and "CREDIT" not in df.columns and "AMOUNT" in df.columns:
        amount_series = df["AMOUNT"].apply(_clean_amount)
        df["DEBIT"] = amount_series.apply(lambda x: abs(x) if x < 0 else 0.0)
        df["CREDIT"] = amount_series.apply(lambda x: x if x > 0 else 0.0)

    if "DEBIT" not in df.columns:
        df["DEBIT"] = 0.0
    if "CREDIT" not in df.columns:
        df["CREDIT"] = 0.0
    if "BALANCE" not in df.columns:
        df["BALANCE"] = 0.0

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DESCRIPTION"] = df["DESCRIPTION"].astype(str).fillna("").str.strip()
    df["DEBIT"] = df["DEBIT"].apply(_clean_amount)
    df["CREDIT"] = df["CREDIT"].apply(_clean_amount)
    df["BALANCE"] = df["BALANCE"].apply(_clean_amount)
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    return df[["DATE", "DESCRIPTION", "DEBIT", "CREDIT", "BALANCE"]]


def parse_bank_statement(file_path: str | Path) -> pd.DataFrame:
    text, dataframe, metadata = load_file(file_path)
    if dataframe is not None:
        return normalize_bank_dataframe(dataframe)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(
            r"(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?P<description>.+?)\s+(?P<debit>-?[\d,]+(?:\.\d+)?)?\s+(?P<credit>-?[\d,]+(?:\.\d+)?)?\s+(?P<balance>-?[\d,]+(?:\.\d+)?)$",
            line,
        )
        if match:
            rows.append(match.groupdict())

    if not rows:
        raise ValueError(
            f"Could not parse bank statement text from {file_path}. Prefer CSV input with standard columns."
        )

    df = pd.DataFrame(rows)
    return normalize_bank_dataframe(df)
