from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ..schemas import LabelRow
from .rules import CATEGORY_RULES, RuleBasedCategoryEngine


ALLOWED_CATEGORIES = set(CATEGORY_RULES.keys()) | {"UNCATEGORIZED"}


class HybridBankStatementLabeler:
    def __init__(self, llm, enable_llm_refinement: bool = True) -> None:
        self.llm = llm
        self.enable_llm_refinement = enable_llm_refinement
        self.rule_engine = RuleBasedCategoryEngine()

    def _refine_with_llm(self, description: str, debit: float, credit: float) -> str | None:
        if not (self.enable_llm_refinement and self.llm and self.llm.available):
            return None
        label = self.llm.classify_transaction(description, debit, credit)
        if label in ALLOWED_CATEGORIES:
            return label
        return None

    def label_dataframe(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, List[LabelRow]]:
        if dataframe.empty:
            empty = dataframe.copy()
            for col in ["CATEGORY", "LABEL_SOURCE", "CONFIDENCE"]:
                empty[col] = []
            return empty, []

        records: List[LabelRow] = []
        rows = []

        for _, row in dataframe.iterrows():
            description = str(row.get("DESCRIPTION", "")).strip()
            debit = float(row.get("DEBIT", 0.0) or 0.0)
            credit = float(row.get("CREDIT", 0.0) or 0.0)
            balance = float(row.get("BALANCE", 0.0) or 0.0)
            date_value = row.get("DATE")
            date_text = pd.to_datetime(date_value).date().isoformat()

            rule_decision = self.rule_engine.classify(description, debit, credit)
            category = rule_decision.category
            source = rule_decision.source
            confidence = rule_decision.confidence

            llm_label = self._refine_with_llm(description, debit, credit)
            if llm_label and (category == "UNCATEGORIZED" or confidence < 0.85):
                category = llm_label
                source = "llm-hybrid"
                confidence = max(confidence, 0.82)

            labeled = LabelRow(
                DATE=date_text,
                DESCRIPTION=description,
                DEBIT=round(debit, 2),
                CREDIT=round(credit, 2),
                BALANCE=round(balance, 2),
                CATEGORY=category,
                LABEL_SOURCE=source,
                CONFIDENCE=round(confidence, 3),
                RAW_ROW={key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in row.to_dict().items()},
            )
            records.append(labeled)
            rows.append(labeled.model_dump())

        labeled_df = pd.DataFrame(rows)
        return labeled_df, records
