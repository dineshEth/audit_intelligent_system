from audit_intelligence.config import get_settings
from audit_intelligence.db import MongoManager


def main() -> None:
    settings = get_settings()
    mongo = MongoManager(settings.mongodb_uri, settings.mongodb_db_name)
    mongo.ensure_indexes()
    print({"ping": mongo.ping(), "db": settings.mongodb_db_name})


if __name__ == "__main__":
    main()
