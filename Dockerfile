FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Pin uv so Docker builds are reproducible.
RUN pip install --no-cache-dir uv==0.12.10

# Install locked runtime dependencies first. This layer is reused when only
# application code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

RUN useradd --create-home --uid 10001 appuser
RUN mkdir -p /app/data/documents && chown appuser:appuser /app/data/documents
USER appuser

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "nexusrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
