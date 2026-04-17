import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_intelligence.config import get_settings
from audit_intelligence.finetune.watcher import DataWatcher
from audit_intelligence.services.pipeline_service import PipelineService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()

    service = PipelineService(get_settings())
    watcher = DataWatcher(service.repositories, service.settings)
    print(f"Watching {service.settings.labeled_data_dir} and {service.settings.qa_data_dir} every {args.interval}s")
    watcher.watch_forever(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
