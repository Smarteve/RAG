# NexusRAG

An async multimodal document processing and retrieval-augmented generation (RAG) engine.

## Run locally

```bash
uv run uvicorn nexusrag.main:app --reload
```

Then open <http://127.0.0.1:8000/docs>.

## Checks and CI

Run the same checks as GitHub Actions locally:

```bash
uv sync --locked --dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests,
using Python from `.python-version`. View results in the repository Actions tab.
The `--locked` option fails if `uv.lock` needs updating, keeping CI dependencies
consistent with the committed lockfile.
