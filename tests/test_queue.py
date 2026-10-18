import asyncio
from unittest.mock import Mock
from uuid import uuid4

from nexusrag.services import queue


def test_enqueue_passes_importable_task_and_string_id(monkeypatch):
    enqueue = Mock()
    monkeypatch.setattr(queue.queue, "enqueue_call", enqueue)
    job_id = uuid4()
    asyncio.run(queue.enqueue_processing(job_id))
    enqueue.assert_called_once_with(
        "nexusrag.worker.process_document", args=(str(job_id),), job_id=str(job_id)
    )
