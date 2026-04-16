from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from ..utils.hashing import sha256_file
from .dataset_builder import FineTuneDatasetBuilder
from .trainer import LocalFineTuneTrainer


class DataWatcher:
    def __init__(self, repositories, settings) -> None:
        self.repositories = repositories
        self.settings = settings
        self.builder = FineTuneDatasetBuilder(repositories, settings)
        self.trainer = LocalFineTuneTrainer(repositories, settings)
        self.manifest_path = self.settings.models_dir / "data_manifest.json"

    def _tracked_files(self) -> List[Path]:
        tracked = []
        for folder in [self.settings.labeled_data_dir, self.settings.qa_data_dir]:
            if folder.exists():
                for path in sorted(folder.glob("**/*")):
                    if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"}:
                        tracked.append(path)
        return tracked

    def current_manifest(self) -> Dict[str, str]:
        return {str(path.relative_to(self.settings.project_root)): sha256_file(path) for path in self._tracked_files()}

    def previous_manifest(self) -> Dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def has_changes(self) -> bool:
        return self.current_manifest() != self.previous_manifest()

    def save_manifest(self, manifest: Dict[str, str]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def scan_and_maybe_finetune(self):
        current = self.current_manifest()
        if current == self.previous_manifest():
            return None

        examples = self.builder.build_examples()
        dataset_path = self.builder.save_dataset(examples)
        result = self.trainer.run(dataset_path)
        self.save_manifest(current)
        return result

    def watch_forever(self, interval_seconds: int = 20) -> None:
        while True:
            self.scan_and_maybe_finetune()
            time.sleep(interval_seconds)
