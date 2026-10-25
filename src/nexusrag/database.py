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
            # Enforce one child per failed attempt and one number per chain.
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS unique_retry_parent "
                "ON processing_jobs(CASE WHEN json_valid(record) "
                "THEN json_extract(record, '$.previous_job_id') END)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS unique_chain_attempt ON processing_jobs("
                "CASE WHEN json_valid(record) THEN "
                "COALESCE(json_extract(record, '$.root_job_id'), job_id) ELSE job_id END, "
                "CASE WHEN json_valid(record) THEN "
                "COALESCE(json_extract(record, '$.attempt_number'), 1) ELSE 1 END)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS unique_active_document ON processing_jobs("
                "CASE WHEN json_valid(record) THEN CASE "
                "WHEN json_extract(record, '$.status') IN ('queued', 'running') "
                "THEN json_extract(record, '$.document_id') END END)"
            )

    finally:
        connection.close()
