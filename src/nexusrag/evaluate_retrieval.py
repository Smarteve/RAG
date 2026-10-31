"""Evaluate a saved document's retrieval; generation and ingestion are not run."""

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from uuid import UUID

from qdrant_client.http.exceptions import ApiException

from nexusrag.evaluation_cases import handbook_cases
from nexusrag.schemas.evaluation import RetrievalDataset, RetrievalEvaluationReport
from nexusrag.services.embedding_runtime import EmbeddingRuntime
from nexusrag.services.evaluation import EvaluationInputError, evaluate_retrieval
from nexusrag.services.processing_lock import DocumentBusyError
from nexusrag.services.vector_backend import QdrantBackend


async def _run(
    document_id: UUID, dataset: RetrievalDataset, top_k: int
) -> RetrievalEvaluationReport:
    backend = QdrantBackend()
    try:
        return await evaluate_retrieval(
            document_id, dataset, EmbeddingRuntime(), backend=backend, top_k=top_k
        )
    finally:
        await backend.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "document_id", type=UUID, help="An existing indexed document ID"
    )
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=5)
    parser.add_argument(
        "--cases",
        type=Path,
        help="Custom dataset JSON; default: bundled handbook cases",
    )
    parser.add_argument(
        "--output", type=Path, help="Save full JSON report and print only its summary"
    )
    args = parser.parse_args(argv)
    try:
        dataset = (
            RetrievalDataset.model_validate_json(args.cases.read_text(encoding="utf-8"))
            if args.cases
            else handbook_cases()
        )
        report = asyncio.run(_run(args.document_id, dataset, args.top_k))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                report.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            summary = report.model_dump(mode="json", exclude={"results"})
            summary["report_path"] = str(args.output)
            print(json.dumps(summary, indent=2))
        else:
            print(report.model_dump_json(indent=2))
    except (EvaluationInputError, DocumentBusyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, sqlite3.Error, RuntimeError, ApiException) as error:
        print(
            f"Evaluation failed ({type(error).__name__}); check the document, dataset, cached model, and Qdrant",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
