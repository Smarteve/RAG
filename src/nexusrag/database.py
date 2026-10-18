import sqlite3
from pathlib import Path


def initialize_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)

    try:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY NOT NULL,
                    record TEXT NOT NULL
                )
                """)

    finally:
        connection.close()
