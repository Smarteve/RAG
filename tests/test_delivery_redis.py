"""Opt-in tests against Redis, using private queues and generated job IDs only."""

import asyncio
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from redis import Redis
from redis.exceptions import TimeoutError
from rq import Queue
from rq.exceptions import DuplicateJobError
from rq.serializers import JSONSerializer

from nexusrag import config
from nexusrag.database import initialize_database
from nexusrag.schemas.job import ProcessingJob
from nexusrag.services.delivery_reconciliation import reconcile_queued_jobs
from nexusrag.services.job_storage import _load_job

REDIS_URL = os.environ.get("NEXUSRAG_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not REDIS_URL,
    reason="Set NEXUSRAG_TEST_REDIS_URL for private-queue Redis integration tests",
)


@pytest.fixture
def real_queue(tmp_path, monkeypatch):
    client = Redis.from_url(REDIS_URL, socket_connect_timeout=5, socket_timeout=5)
    client.ping()
    queue = Queue(
        f"nexusrag-test-delivery-{uuid4().hex}",
        connection=client,
        serializer=JSONSerializer,
    )
    path = tmp_path / "jobs.sqlite3"
    initialize_database(path)
    monkeypatch.setattr(config, "STORAGE_DIRECTORY", tmp_path / "documents")
    config.STORAGE_DIRECTORY.mkdir()
    now = datetime.now(UTC) - timedelta(minutes=5)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO processing_jobs VALUES (?, ?)",
                (str(job.job_id), job.model_dump_json()),
            )
    finally:
        connection.close()
    (config.STORAGE_DIRECTORY / f"{job.document_id}.pdf").write_bytes(b"%PDF-saved")
    try:
        yield queue, path, job
    finally:
        queue.delete(delete_jobs=True)
        client.close()


def test_real_rq_unique_enqueue_is_atomic(real_queue):
    queue, _, job = real_queue

    def enqueue(_):
        try:
            queue.enqueue_call(
                "nexusrag.tasks.process_document",
                args=(str(job.job_id),),
                job_id=str(job.job_id),
                unique=True,
            )
            return True
        except DuplicateJobError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(enqueue, range(2))) == [False, True]
    assert queue.job_ids == [str(job.job_id)]


def test_real_concurrent_sweeps_preserve_one_delivery(real_queue):
    queue, path, job = real_queue

    def sweep(_):
        return asyncio.run(
            reconcile_queued_jobs(path, REDIS_URL, apply=True, queue_name=queue.name)
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(pool.map(sweep, range(2)))
    assert sum(d.action == "enqueued" for d in decisions) == 1
    assert queue.job_ids == [str(job.job_id)]
    assert _load_job(job.job_id, path) == job


def test_real_success_with_lost_ack_is_not_enqueued_again(real_queue, monkeypatch):
    queue, path, job = real_queue
    original = Queue.enqueue_call

    def lost_ack(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise TimeoutError("Simulated lost acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(Queue, "enqueue_call", lost_ack)
        first = asyncio.run(
            reconcile_queued_jobs(path, REDIS_URL, apply=True, queue_name=queue.name)
        )
    assert first[0].action == "delivery_unconfirmed"
    second = asyncio.run(
        reconcile_queued_jobs(path, REDIS_URL, apply=True, queue_name=queue.name)
    )
    assert second[0].action == "present"
    assert queue.job_ids == [str(job.job_id)]
    assert _load_job(job.job_id, path) == job
