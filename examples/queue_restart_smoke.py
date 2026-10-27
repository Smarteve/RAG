"""Opt-in real Redis restart/whole-worker crash smoke test.

Uses the existing redis:8-alpine image (--pull=never), a private container/volume,
private queue, temporary SQLite/PDF/index paths, and the existing cached MiniLM.
No Groq calls, model downloads, or production Redis restarts. Run from repo root:
    uv run --no-sync python -m examples.queue_restart_smoke
"""

import atexit
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import UUID, uuid4


def paused_task(job_id: str) -> None:
    """Test task: establish real ownership, then wait for deliberate SIGKILL."""
    from nexusrag import config
    from nexusrag.services.job_storage import _load_job, _transition_job
    from nexusrag.services.processing_lock import document_lock

    identifier = UUID(job_id)
    job = _load_job(identifier, config.JOB_DATABASE_PATH)
    with document_lock(job.document_id, config.JOB_DATABASE_PATH, blocking=True):
        _transition_job(identifier, "running", config.JOB_DATABASE_PATH)
        (config.JOB_DATABASE_PATH.parent / "child.pid").write_text(str(os.getpid()))
        while True:
            time.sleep(1)


def run_worker() -> None:
    # This walkthrough targets queue/crash behavior, not server persistence.
    # Use the real adapter with an isolated in-memory client in each workhorse.
    from qdrant_client import AsyncQdrantClient
    from redis import Redis
    from rq import Queue, Worker
    from rq.serializers import JSONSerializer

    from nexusrag import config, tasks
    from nexusrag.services.vector_backend import QdrantBackend
    from nexusrag.worker_hooks import handle_job_exception, handle_workhorse_killed

    class SmokeVectorBackend(QdrantBackend):
        def __init__(self):
            super().__init__(AsyncQdrantClient(":memory:"), collection="restart-smoke")

        async def aclose(self):
            await self.client.close()

    tasks.QdrantBackend = SmokeVectorBackend
    connection = Redis.from_url(config.REDIS_URL)
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
            maintenance_interval=1,
            job_monitoring_interval=1,
        )
        worker.work(burst=True)
    finally:
        connection.close()


def wait_for(check, timeout: int = 90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.2)
    raise TimeoutError("Smoke test condition did not complete")


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True).strip()


def exercise(directory: Path, container: str) -> None:
    import asyncio
    import shutil
    from datetime import UTC, datetime

    from redis import Redis
    from redis.backoff import NoBackoff
    from redis.exceptions import RedisError
    from redis.retry import Retry
    from rq import Queue
    from rq.job import Job
    from rq.serializers import JSONSerializer

    port = docker("port", container, "6379/tcp").rsplit(":", 1)[1]
    root = Path.cwd()
    os.environ.update(
        {
            "NEXUSRAG_REDIS_URL": f"redis://127.0.0.1:{port}/0",
            "NEXUSRAG_INGESTION_QUEUE_NAME": "restart-smoke",
            "NEXUSRAG_JOB_DATABASE_PATH": str(directory / "jobs.sqlite3"),
            "NEXUSRAG_STORAGE_DIR": str(directory / "documents"),
            "NEXUSRAG_EXTRACTION_DIR": str(directory / "extracted"),
            "NEXUSRAG_CHUNK_DIR": str(directory / "chunks"),
            "NEXUSRAG_EMBEDDING_DIR": str(directory / "embeddings"),
            "NEXUSRAG_MODEL_CACHE_DIR": str(root / "data/models"),
        }
    )
    # Import application configuration only after isolating all paths.
    from nexusrag import config
    from nexusrag.database import initialize_database
    from nexusrag.schemas.job import ProcessingJob
    from nexusrag.services.job_storage import _create_job, _load_document_job, _load_job
    from nexusrag.storage import load_embeddings
    from nexusrag.sweep_jobs import sweep_once

    connection = Redis.from_url(
        config.REDIS_URL,
        socket_connect_timeout=2,
        socket_timeout=2,
        retry=Retry(NoBackoff(), 0),
    )

    def ping():
        try:
            return connection.ping()
        except RedisError:
            return False

    wait_for(ping)
    initialize_database(config.JOB_DATABASE_PATH)
    now = datetime.now(UTC)
    job = ProcessingJob(
        job_id=uuid4(),
        document_id=uuid4(),
        status="queued",
        created_at=now,
        updated_at=now,
    )
    config.STORAGE_DIRECTORY.mkdir()
    shutil.copyfile(
        root / "examples/employee-handbook.pdf",
        config.STORAGE_DIRECTORY / f"{job.document_id}.pdf",
    )
    _create_job(job, config.JOB_DATABASE_PATH)
    queue = Queue(
        config.INGESTION_QUEUE_NAME, connection=connection, serializer=JSONSerializer
    )
    queue.enqueue_call(
        "examples.queue_restart_smoke.paused_task",
        args=(str(job.job_id),),
        job_id=str(job.job_id),
        unique=True,
        timeout=180,
    )
    time.sleep(1.2)  # Allow the configured once-per-second AOF flush.
    docker("restart", container)
    wait_for(ping)
    assert queue.job_ids == [str(job.job_id)]
    print("PASS: accepted queued task survived broker container restart", flush=True)

    command = [sys.executable, "-m", "examples.queue_restart_smoke", "--worker"]
    with (directory / "worker.log").open("w+") as log:
        worker = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_for(
                lambda: (
                    _load_job(job.job_id, config.JOB_DATABASE_PATH).status == "running"
                ),
                timeout=30,
            )
            marker = directory / "child.pid"
            wait_for(marker.exists, timeout=10)
            child_pid = int(marker.read_text())
            child_start = (
                Path(f"/proc/{child_pid}/stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19]
            )

            def cleanup_child():
                try:
                    current_start = (
                        Path(f"/proc/{child_pid}/stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()[19]
                    )
                    if current_start == child_start:
                        os.kill(child_pid, signal.SIGKILL)
                except FileNotFoundError, ProcessLookupError:
                    pass

            atexit.register(cleanup_child)
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=10)
        assert _load_job(job.job_id, config.JOB_DATABASE_PATH).status == "running"
        rq_job = Job.fetch(
            str(job.job_id), connection=connection, serializer=JSONSerializer
        )
        executions = rq_job.get_executions()
        assert executions
        # Let real registry deadlines expire; don't edit Redis to simulate time.
        deadline = max(
            max(
                connection.zscore(
                    rq_job.started_job_registry.key, execution.composite_key
                ),
                connection.zscore(rq_job.execution_registry.key, execution.id),
            )
            for execution in executions
        )
        delay = max(0, deadline - time.time()) + 2
        print(
            f"PASS: whole worker killed; waiting {delay:.0f}s for real RQ abandonment expiry",
            flush=True,
        )
        time.sleep(delay)
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30
        )
        assert (
            Job.fetch(str(job.job_id), connection=connection).get_status().value
            == "failed"
        )
        assert _load_job(job.job_id, config.JOB_DATABASE_PATH).status == "running"
        print(
            "PASS: restarted worker detected abandonment; SQLite left for guarded recovery",
            flush=True,
        )
        # Parent loss must not revoke a surviving task's ownership.
        assert asyncio.run(sweep_once(apply=True, max_attempts=3)) == 0
        assert _load_job(job.job_id, config.JOB_DATABASE_PATH).status == "running"
        print(
            "PASS: orphan child retained ownership; recovery correctly skipped it",
            flush=True,
        )
        cleanup_child()
        atexit.unregister(cleanup_child)
        from nexusrag.services.processing_lock import DocumentBusyError, document_lock

        def lock_released():
            try:
                with document_lock(job.document_id, config.JOB_DATABASE_PATH):
                    return True
            except DocumentBusyError:
                return False

        wait_for(lock_released, timeout=10)
        assert asyncio.run(sweep_once(apply=True, max_attempts=3)) == 0
        retry = _load_document_job(job.document_id, config.JOB_DATABASE_PATH)
        assert retry.attempt_number == 2 and retry.status == "queued"
        assert _load_job(job.job_id, config.JOB_DATABASE_PATH).status == "failed"
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
        )
        completed = _load_job(retry.job_id, config.JOB_DATABASE_PATH)
        if result.returncode != 0 or completed.status != "succeeded":
            log.seek(0)
            raise AssertionError(log.read())
        embeddings = asyncio.run(
            load_embeddings(job.document_id, config.EMBEDDING_DIRECTORY)
        )
        assert embeddings.embeddings
        print(
            f"PASS: sweep recovered and retried; real cached MiniLM saved {len(embeddings.embeddings)} vectors",
            flush=True,
        )
    connection.close()


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        run_worker()
        return
    token = uuid4().hex[:12]
    container = f"nexusrag-restart-test-{token}"
    volume = f"nexusrag-restart-test-{token}"
    docker("image", "inspect", "redis:8-alpine")
    docker("volume", "create", volume)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    try:
        docker(
            "run",
            "--pull=never",
            "-d",
            "--name",
            container,
            "-p",
            f"127.0.0.1:{port}:6379",
            "-v",
            f"{volume}:/data",
            "redis:8-alpine",
            "redis-server",
            "--appendonly",
            "yes",
            "--appendfsync",
            "everysec",
        )
        with tempfile.TemporaryDirectory(prefix="nexusrag-restart-") as directory:
            exercise(Path(directory), container)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], check=False, capture_output=True
        )
        subprocess.run(
            ["docker", "volume", "rm", volume], check=False, capture_output=True
        )


if __name__ == "__main__":
    main()
