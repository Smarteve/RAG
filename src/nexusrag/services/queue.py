from uuid import UUID

from fastapi.concurrency import run_in_threadpool
from redis import Redis
from rq import Queue
from rq.serializers import JSONSerializer

from nexusrag.config import INGESTION_QUEUE_NAME, REDIS_URL  # Python client package

client = Redis.from_url(
    REDIS_URL,
    socket_connect_timeout=5,
    socket_timeout=5,
)

queue = Queue(
    INGESTION_QUEUE_NAME,
    connection=client,
    serializer=JSONSerializer,
)


async def enqueue_processing(job_id: UUID) -> None:
    await run_in_threadpool(
        queue.enqueue_call,
        "nexusrag.tasks.process_document",
        args=(str(job_id),),
        job_id=str(job_id),
    )
