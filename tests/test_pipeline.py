from pathlib import Path

from audit_intelligence.config import Settings
from audit_intelligence.services.pipeline_service import PipelineService


def test_pipeline_generates_outputs(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        mongodb_uri="mongomock://localhost",
        mongodb_db_name="test_db",
        use_faiss=False,
    )
    settings.ensure_directories()

    sample_src = Path(__file__).resolve().parents[1] / "datasets" / "raw_samples" / "sample_bank_statement.csv"
    sample_dst = tmp_path / "sample_bank_statement.csv"
    sample_dst.write_text(sample_src.read_text(encoding="utf-8"), encoding="utf-8")

    service = PipelineService(settings)
    result = service.process_file(sample_dst)

    response = result["response"]
    assert response["report_path"]
    assert Path(response["report_path"]).exists()
    assert response["labels"]["export_paths"]["csv"]
    assert Path(response["labels"]["export_paths"]["csv"]).exists()
    assert response["qa_pair_count"] > 0
