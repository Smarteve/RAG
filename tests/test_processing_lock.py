import subprocess
import sys
from uuid import uuid4

import pytest

from nexusrag.services.processing_lock import DocumentBusyError, document_lock


def test_document_lock_excludes_other_owners_and_releases_on_error(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    document_id = uuid4()
    with (
        pytest.raises(RuntimeError, match="processing error"),
        document_lock(document_id, path),
    ):
        with pytest.raises(DocumentBusyError), document_lock(document_id, path):
            pytest.fail("Second owner acquired document")
        with document_lock(uuid4(), path):
            pass
        raise RuntimeError("processing error")
    with document_lock(document_id, path):
        pass


def test_lock_survives_while_other_process_lives_and_releases_on_death(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    document_id = uuid4()
    script = """
import sys
from pathlib import Path
from uuid import UUID
from nexusrag.services.processing_lock import document_lock
with document_lock(UUID(sys.argv[1]), Path(sys.argv[2])):
    print('locked', flush=True)
    sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(document_id), str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(DocumentBusyError), document_lock(document_id, path):
            pytest.fail("Acquired a live process's lock")
        child.kill()
        child.wait(timeout=5)
        with document_lock(document_id, path):
            pass
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
