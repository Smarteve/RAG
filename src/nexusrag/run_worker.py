"""Start a JSON RQ worker with application failure recovery hooks."""

from redis import Redis
from rq import Queue, Worker
from rq.serializers import JSONSerializer

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.worker_hooks import handle_job_exception, handle_workhorse_killed


def main() -> None:
    initialize_database(config.JOB_DATABASE_PATH)
    connection = Redis.from_url(
        config.REDIS_URL, socket_connect_timeout=5, socket_timeout=5
    )
    try:
        queue = Queue(
            config.INGESTION_QUEUE_NAME,
            connection=connection,
            serializer=JSONSerializer,
        )
        worker = Worker(
            [queue],
            connection=connection,
            serializer=JSONSerializer,
            exception_handlers=[handle_job_exception],
            work_horse_killed_handler=handle_workhorse_killed,
        )
        worker.work()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
