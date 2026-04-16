from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pypdf import PdfReader

from ..utils.text import normalize_whitespace


def infer_doc_type(text: str, dataframe: Optional[pd.DataFrame] = None) -> str:
    if dataframe is not None:
        lowered = {str(col).strip().lower() for col in dataframe.columns}
        bank_markers = {"date", "description", "debit", "credit", "balance", "amount"}
        if len(lowered.intersection(bank_markers)) >= 3:
            return "bank_statement"
    lowered_text = (text or "").lower()
    if any(term in lowered_text for term in ["account statement", "transaction", "debit", "credit", "balance"]):
        return "bank_statement"
    return "generic"

def load_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def load_file(file_path: str | Path) -> Tuple[str, Optional[pd.DataFrame], Dict[str, Any]]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    dataframe: Optional[pd.DataFrame] = None

    if suffix == ".csv":
        dataframe = pd.read_csv(path)
        text = dataframe.to_csv(index=False)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            dataframe = pd.DataFrame(payload)
            text = dataframe.to_csv(index=False)
        else:
            text = json.dumps(payload, indent=2)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        text = load_pdf_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    text = normalize_whitespace(text)
    mime_type, _ = mimetypes.guess_type(path.name)
    metadata = {
        "file_name": path.name,
        "file_path": str(path),
        "mime_type": mime_type or "text/plain",
        "doc_type": infer_doc_type(text, dataframe),
        "row_count": int(dataframe.shape[0]) if dataframe is not None else None,
        "column_count": int(dataframe.shape[1]) if dataframe is not None else None,
    }
    return text, dataframe, metadata
