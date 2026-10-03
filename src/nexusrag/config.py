import os
from pathlib import Path

# Relative paths are resolved from the application's working directory.
STORAGE_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_STORAGE_DIR", "data/documents"))
    .expanduser()
    .resolve()
)


EXTRACTION_DIRECTORY = (
    Path(os.environ.get("NEXUSRAG_EXTRACTION_DIR", "data/extracted"))
    .expanduser()
    .resolve()
)
