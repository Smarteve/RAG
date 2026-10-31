import asyncio
import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from nexusrag import evaluate_retrieval as cli
from nexusrag.schemas.evaluation import RetrievalEvaluationReport
from nexusrag.services.retrieval import StaleSemanticIndexError


def report(document_id):
    return RetrievalEvaluationReport(
        dataset_id="test",
        document_id=document_id,
        document_sha256="a" * 64,
        source_digest="b" * 64,
        index_generation=uuid4(),
        model_id="test",
        model_revision="v1",
        chunk_size=1000,
        overlap=200,
        chunk_count=2,
        top_k=5,
        answerable_cases=1,
        unsupported_cases=0,
        hit_rate_at_k=0.0,
        mrr_at_k=0.0,
        results=[],
    )


def test_cli_saves_report_and_returns_success_even_when_retrieval_misses(
    tmp_path, monkeypatch, capsys
):
    document_id = uuid4()
    expected = report(document_id)
    run = AsyncMock(return_value=expected)
    monkeypatch.setattr(cli, "_run", run)
    destination = tmp_path / "reports" / "evaluation.json"
    assert cli.main([str(document_id), "--output", str(destination)]) == 0
    assert (
        RetrievalEvaluationReport.model_validate_json(destination.read_text())
        == expected
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["hit_rate_at_k"] == 0.0
    assert "results" not in summary
    assert summary["report_path"] == str(destination)
    assert run.await_args.args[1].dataset_id == "employee-handbook-v1"


def test_cli_failures_do_not_write_partial_reports_or_expose_provider_details(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "_run", AsyncMock(side_effect=ConnectionError("private key and URL"))
    )
    destination = tmp_path / "report.json"
    assert cli.main([str(uuid4()), "--output", str(destination)]) == 1
    assert not destination.exists()
    assert "private" not in capsys.readouterr().err


def test_cli_prints_full_report_without_an_output_file(monkeypatch, capsys):
    document_id = uuid4()
    expected = report(document_id)
    monkeypatch.setattr(cli, "_run", AsyncMock(return_value=expected))
    assert cli.main([str(document_id)]) == 0
    assert "results" in json.loads(capsys.readouterr().out)


def test_cli_stale_index_reports_rebuild_endpoint_without_writing_scores(
    tmp_path, monkeypatch, capsys
):
    document_id = uuid4()
    monkeypatch.setattr(
        cli, "_run", AsyncMock(side_effect=StaleSemanticIndexError("internal details"))
    )
    destination = tmp_path / "report.json"
    assert cli.main([str(document_id), "--output", str(destination)]) == 1
    captured = capsys.readouterr()
    assert f"POST /documents/{document_id}/semantic-index" in captured.err
    assert "then rerun evaluation" in captured.err
    assert "internal details" not in captured.err
    assert not captured.out
    assert not destination.exists()


@pytest.mark.parametrize("value", ["0", "21"])
def test_cli_rejects_invalid_limit(value):
    with pytest.raises(SystemExit):
        cli.main([str(uuid4()), "--top-k", value])


def test_runner_closes_backend_after_evaluation_failure(monkeypatch):
    backend = Mock(aclose=AsyncMock())
    monkeypatch.setattr(cli, "QdrantBackend", lambda: backend)
    monkeypatch.setattr(
        cli, "evaluate_retrieval", AsyncMock(side_effect=RuntimeError("failed"))
    )
    with pytest.raises(RuntimeError):
        asyncio.run(cli._run(uuid4(), cli.handbook_cases(), 5))
    backend.aclose.assert_awaited_once_with()
