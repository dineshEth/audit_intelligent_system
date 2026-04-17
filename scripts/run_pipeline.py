import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_intelligence.config import get_settings
from audit_intelligence.services.pipeline_service import PipelineService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to a local file.")
    parser.add_argument(
        "--query",
        default="Analyze this document for audit review, label bank transactions if applicable, and produce a report.",
    )
    args = parser.parse_args()

    service = PipelineService(get_settings())
    result = service.process_file(Path(args.file), user_query=args.query)
    print(json.dumps(result["response"], indent=2, default=str))


if __name__ == "__main__":
    main()
