FROM python:3.15.0-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Pin uv so Docker builds are reproducible.
RUN pip install --no-cache-dir uv==0.13.0

# Some dependencies do not yet publish Python 3.15 wheels. Keep their
# build tools in this stage, out of the final application image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install locked runtime dependencies first. This layer is reused when only
# application code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.15.0-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app /app

RUN useradd --create-home --uid 10001 appuser
RUN mkdir -p /app/data/documents /app/data/extracted /app/data/chunks /app/data/embeddings \
    && chown appuser:appuser /app/data /app/data/documents /app/data/extracted /app/data/chunks /app/data/embeddings
USER appuser

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "nexusrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
