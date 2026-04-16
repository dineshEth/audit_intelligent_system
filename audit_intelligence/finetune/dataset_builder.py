from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..utils.dates import utcnow_iso


class FineTuneDatasetBuilder:
    def __init__(self, repositories, settings) -> None:
        self.repositories = repositories
        self.settings = settings

    def build_examples(self, max_qa: int = 500, max_labels: int = 1000) -> List[Dict[str, str]]:
        examples: List[Dict[str, str]] = []

        qa_records = self.repositories.qa_pairs.find_many({}, limit=max_qa, sort=("created_at", -1))
        for record in qa_records:
            examples.append(
                {
                    "instruction": record.question.strip(),
                    "output": record.answer.strip(),
                    "source": record.source_type,
                    "document_id": record.document_id or "",
                }
            )

        label_records = self.repositories.labels.find_many({}, limit=100, sort=("created_at", -1))
        count = 0
        for record in label_records:
            for item in record.labels:
                examples.append(
                    {
                        "instruction": f"Classify the bank transaction into a category: {item.DESCRIPTION}",
                        "output": item.CATEGORY,
                        "source": "label-derived",
                        "document_id": record.document_id,
                    }
                )
                count += 1
                if count >= max_labels:
                    break
            if count >= max_labels:
                break

        # Deduplicate by instruction/output pair
        deduped = {}
        for example in examples:
            key = (example["instruction"], example["output"])
            deduped[key] = example
        return list(deduped.values())

    def save_dataset(self, examples: List[Dict[str, str]]) -> Path:
        path = self.settings.qa_data_dir / f"finetune_dataset_{utcnow_iso()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in examples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
