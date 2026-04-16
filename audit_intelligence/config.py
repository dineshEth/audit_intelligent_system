from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    mongodb_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    mongodb_db_name: str = field(default_factory=lambda: os.getenv("MONGODB_DB_NAME", "audit_intelligence"))
    use_faiss: bool = field(default_factory=lambda: _as_bool(os.getenv("USE_FAISS"), False))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")))
    max_retry: int = field(default_factory=lambda: int(os.getenv("MAX_RETRY", "1")))
    local_llm_path: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_PATH", "").strip())
    local_embedding_model_path: str = field(default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "").strip())
    base_finetune_model_path: str = field(default_factory=lambda: os.getenv("BASE_FINETUNE_MODEL_PATH", "").strip())
    finetune_output_dir: str = field(default_factory=lambda: os.getenv("FINETUNE_OUTPUT_DIR", "outputs/models"))
    finetune_max_length: int = field(default_factory=lambda: int(os.getenv("FINETUNE_MAX_LENGTH", "384")))
    finetune_epochs: int = field(default_factory=lambda: int(os.getenv("FINETUNE_EPOCHS", "1")))
    finetune_batch_size: int = field(default_factory=lambda: int(os.getenv("FINETUNE_BATCH_SIZE", "2")))
    finetune_learning_rate: float = field(default_factory=lambda: float(os.getenv("FINETUNE_LEARNING_RATE", "2e-4")))
    report_author: str = field(default_factory=lambda: os.getenv("REPORT_AUTHOR", "Audit Intelligence System"))
    enable_llm_label_refinement: bool = field(default_factory=lambda: _as_bool(os.getenv("ENABLE_LLM_LABEL_REFINEMENT"), True))

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def raw_samples_dir(self) -> Path:
        return self.datasets_dir / "raw_samples"

    @property
    def labeled_data_dir(self) -> Path:
        return self.datasets_dir / "labeled_data"

    @property
    def qa_data_dir(self) -> Path:
        return self.datasets_dir / "qa_data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def charts_dir(self) -> Path:
        return self.outputs_dir / "charts"

    @property
    def labeled_docs_dir(self) -> Path:
        return self.outputs_dir / "labeled_docs"

    @property
    def models_dir(self) -> Path:
        out = Path(self.finetune_output_dir)
        if not out.is_absolute():
            out = self.project_root / out
        return out

    @property
    def upload_dir(self) -> Path:
        return self.project_root / "uploads"

    def ensure_directories(self) -> None:
        for path in self.iter_directories():
            path.mkdir(parents=True, exist_ok=True)

    def iter_directories(self) -> Iterable[Path]:
        return [
            self.datasets_dir,
            self.raw_samples_dir,
            self.labeled_data_dir,
            self.qa_data_dir,
            self.outputs_dir,
            self.reports_dir,
            self.charts_dir,
            self.labeled_docs_dir,
            self.models_dir,
            self.upload_dir,
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
