from pathlib import Path

from audit_intelligence.config import get_settings
from audit_intelligence.services.pipeline_service import PipelineService


def main() -> None:
    settings = get_settings()
    service = PipelineService(settings)
    sample = settings.raw_samples_dir / "sample_bank_statement.csv"
    result = service.process_file(sample)
    print("Processed:", sample)
    print("Summary:", result["response"].get("summary_text", ""))
    print("Report:", result["response"].get("report_path"))
    print("Labels:", result["response"].get("labels", {}))


if __name__ == "__main__":
    main()
