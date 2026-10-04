from pathlib import Path

import pytest

from nexusrag import config


@pytest.fixture(autouse=True)
def chunk_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep upload tests from writing chunks into real application storage."""
    directory = tmp_path / "chunks"
    monkeypatch.setattr(config, "CHUNK_DIRECTORY", directory)
    return directory
