from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .bank_statement import parse_bank_statement
from .loaders import load_file


def parse_document(file_path: str | Path) -> Dict[str, Any]:
    text, dataframe, metadata = load_file(file_path)
    payload: Dict[str, Any] = {
        "text": text,
        "dataframe": dataframe,
        "metadata": metadata,
        "structured": {},
    }
    if metadata.get("doc_type") == "bank_statement":
        try:
            bank_df = parse_bank_statement(file_path)
            payload["structured"]["transactions"] = bank_df
        except Exception as exc:
            payload["structured"]["bank_parse_error"] = str(exc)
    return payload
