from pathlib import Path
from runpy import run_path
from unittest.mock import Mock

import pytest

# The demo is a standalone script, not part of the installed nexusrag package.
# run_path leaves its main guard inactive, so tests never load/download a model.
_demo = run_path(
    str(Path(__file__).resolve().parents[1] / "examples" / "minilm_smoke.py")
)
MODEL_REVISION = _demo["MODEL_REVISION"]
check_token_lengths = _demo["check_token_lengths"]


def test_token_check_includes_special_tokens_and_disables_truncation() -> None:
    tokenizer = Mock(return_value={"input_ids": [[101, 5, 102], [101, 6, 7, 102]]})

    assert check_token_lengths(tokenizer, ["query", "passage"], max_tokens=256) == [
        3,
        4,
    ]
    tokenizer.assert_called_once_with(
        ["query", "passage"], add_special_tokens=True, truncation=False
    )


def test_token_check_accepts_input_exactly_at_limit() -> None:
    tokenizer = Mock(return_value={"input_ids": [[1] * 256]})

    assert check_token_lengths(tokenizer, ["text"], max_tokens=256) == [256]


def test_token_check_rejects_oversized_input_before_encoding() -> None:
    tokenizer = Mock(return_value={"input_ids": [[1] * 10, [1] * 257]})

    with pytest.raises(ValueError, match="Input 1 has 257 tokens; model limit is 256"):
        check_token_lengths(tokenizer, ["query", "long passage"], max_tokens=256)


def test_model_revision_is_an_immutable_commit_id() -> None:
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)
