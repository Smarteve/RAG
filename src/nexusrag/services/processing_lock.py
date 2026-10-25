"""Cross-process document exclusion on the supported local Linux filesystem.

All task execution and recovery must use the same database path/lock directory.
Do not delete lock files: unlinking them could allow two independent locks.
"""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID


class DocumentBusyError(RuntimeError):
    """Another process owns document processing or recovery."""


@contextmanager
def document_lock(
    document_id: UUID, database_path: Path, *, blocking: bool = False
) -> Iterator[None]:
    directory = database_path.resolve().parent / ".job-locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{document_id}.lock").open("a") as lock:
        try:
            fcntl.flock(
                lock.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            )
        except BlockingIOError as error:
            raise DocumentBusyError("Document execution is still owned") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
