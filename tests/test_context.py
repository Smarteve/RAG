from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nexusrag.schemas.answer import AnswerSource
from nexusrag.schemas.chunk import DocumentChunk
from nexusrag.schemas.retrieval import SemanticSearchHit
from nexusrag.services.context import PreparedContext, build_context


def make_hit(
    text: str,
    *,
    document_id: UUID | None = None,
    chunk_index: int = 0,
    page_number: int = 1,
    similarity: float = 0.5,
) -> SemanticSearchHit:
    return SemanticSearchHit(
        chunk=DocumentChunk(
            document_id=document_id if document_id is not None else uuid4(),
            page_number=page_number,
            chunk_index=chunk_index,
            start_char=0,
            end_char=len(text),
            text=text,
        ),
        similarity=similarity,
    )


def block_for(hit: SemanticSearchHit, source_id: str) -> str:
    return f"[{source_id}]\nPage: {hit.chunk.page_number}\nText:\n{hit.chunk.text}"


def fake_count_tokens(text: str) -> int:
    """Word counts are a deterministic test substitute, not an LLM tokenizer."""
    return len(text.split())


def test_single_source_preserves_original_text_and_page_metadata() -> None:
    hit = make_hit("  Employees receive 25 days of annual leave.\n", page_number=7)

    context = build_context([hit], max_tokens=100, count_tokens=fake_count_tokens)

    assert isinstance(context, PreparedContext)
    assert context.text == block_for(hit, "S1")
    assert context.sources == {
        "S1": AnswerSource(
            source_id="S1",
            document_id=hit.chunk.document_id,
            page_number=7,
            chunk_index=0,
        )
    }


def test_multiple_hits_keep_retrieval_order_not_page_or_score_order() -> None:
    document_id = uuid4()
    hits = [
        make_hit(
            "Work remotely.",
            document_id=document_id,
            chunk_index=3,
            page_number=4,
            similarity=0.2,
        ),
        make_hit(
            "Annual leave.",
            document_id=document_id,
            chunk_index=0,
            page_number=1,
            similarity=0.9,
        ),
    ]

    context = build_context(hits, max_tokens=100, count_tokens=fake_count_tokens)

    assert context.text == "\n\n".join(
        [block_for(hits[0], "S1"), block_for(hits[1], "S2")]
    )
    assert list(context.sources) == ["S1", "S2"]
    assert [source.chunk_index for source in context.sources.values()] == [3, 0]


def test_empty_hits_return_empty_context_without_counting() -> None:
    counter = Mock(side_effect=AssertionError("No sources to count"))

    context = build_context([], max_tokens=1, count_tokens=counter)

    assert context == PreparedContext(text="", sources={})
    counter.assert_not_called()


@pytest.mark.parametrize("max_tokens", [0, -1, True, False, 1.5, "10", None])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_budget_is_rejected_even_for_empty_hits(
    max_tokens, empty: bool
) -> None:
    counter = Mock()
    hits = [] if empty else [make_hit("Hello")]

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        build_context(hits, max_tokens=max_tokens, count_tokens=counter)

    counter.assert_not_called()


def test_exact_budget_includes_block_and_one_token_less_skips_it() -> None:
    hit = make_hit("Employees receive 25 days of annual leave.")
    required = fake_count_tokens(block_for(hit, "S1"))

    exact = build_context([hit], max_tokens=required, count_tokens=fake_count_tokens)
    too_small = build_context(
        [hit], max_tokens=required - 1, count_tokens=fake_count_tokens
    )

    assert exact.text == block_for(hit, "S1")
    assert list(exact.sources) == ["S1"]
    assert too_small == PreparedContext(text="", sources={})


def test_budget_counts_labels_headers_and_separators_not_just_passage_text() -> None:
    document_id = uuid4()
    hits = [
        make_hit("A", document_id=document_id),
        make_hit("B", document_id=document_id, chunk_index=1),
    ]
    first_block = block_for(hits[0], "S1")
    candidate = first_block + "\n\n" + block_for(hits[1], "S2")
    counter = Mock(side_effect=len)
    # With one character less, the second block must be excluded. Counting the
    # two blocks separately would miss the two separator characters and include it.
    context = build_context(hits, max_tokens=len(candidate) - 1, count_tokens=counter)

    assert context.text == first_block
    assert list(context.sources) == ["S1"]
    assert [call.args[0] for call in counter.call_args_list] == [first_block, candidate]


def test_skipped_first_block_does_not_consume_label_or_stop_later_hits() -> None:
    document_id = uuid4()
    hits = [
        make_hit("Long " * 100, document_id=document_id),
        make_hit("Short", document_id=document_id, chunk_index=1, page_number=2),
    ]
    budget = fake_count_tokens(block_for(hits[1], "S1"))

    context = build_context(hits, max_tokens=budget, count_tokens=fake_count_tokens)

    assert context.text == block_for(hits[1], "S1")
    assert context.sources["S1"].chunk_index == 1
    assert context.sources["S1"].page_number == 2


def test_skipped_middle_block_has_no_source_and_labels_have_no_gaps() -> None:
    document_id = uuid4()
    hits = [
        make_hit("First", document_id=document_id),
        make_hit("Long " * 100, document_id=document_id, chunk_index=1),
        make_hit("Last", document_id=document_id, chunk_index=2, page_number=2),
    ]
    expected = block_for(hits[0], "S1") + "\n\n" + block_for(hits[2], "S2")

    context = build_context(
        hits, max_tokens=fake_count_tokens(expected), count_tokens=fake_count_tokens
    )

    assert context.text == expected
    assert list(context.sources) == ["S1", "S2"]
    assert [source.chunk_index for source in context.sources.values()] == [0, 2]


def test_all_oversized_hits_return_empty_context_without_truncation() -> None:
    hits = [make_hit("Long " * 100), make_hit("Another " * 100)]

    assert build_context(
        hits, max_tokens=1, count_tokens=fake_count_tokens
    ) == PreparedContext(text="", sources={})


def test_repeated_chunk_is_included_once_with_first_hit_metadata() -> None:
    hit = make_hit("Original source", chunk_index=5, page_number=3)
    repeated = hit.model_copy(update={"similarity": 0.1})
    counter = Mock(side_effect=fake_count_tokens)

    context = build_context([hit, repeated], max_tokens=100, count_tokens=counter)

    assert context.text == block_for(hit, "S1")
    assert list(context.sources) == ["S1"]
    counter.assert_called_once()


def test_identical_text_or_indices_in_different_documents_are_distinct_sources() -> (
    None
):
    hits = [make_hit("Same text"), make_hit("Same text")]

    context = build_context(hits, max_tokens=100, count_tokens=fake_count_tokens)

    assert list(context.sources) == ["S1", "S2"]
    assert [source.document_id for source in context.sources.values()] == [
        hit.chunk.document_id for hit in hits
    ]


@pytest.mark.parametrize(
    "count", [-1, 1.5, True, False, "10", None, float("nan"), float("inf")]
)
def test_invalid_token_counter_result_is_rejected(count) -> None:
    with pytest.raises(
        ValueError, match="count_tokens must return a nonnegative integer"
    ):
        build_context([make_hit("Hello")], max_tokens=100, count_tokens=lambda _: count)


def test_tokenizer_error_propagates_without_mutating_inputs() -> None:
    hits = [make_hit("Hello")]
    original = [hit.model_dump() for hit in hits]
    error = RuntimeError("Tokenizer unavailable")

    with pytest.raises(RuntimeError) as raised:
        build_context(hits, max_tokens=100, count_tokens=Mock(side_effect=error))

    assert raised.value is error
    assert [hit.model_dump() for hit in hits] == original


@given(
    texts=st.lists(st.text(min_size=1, max_size=80), max_size=8),
    budget=st.integers(min_value=1, max_value=800),
)
def test_generated_context_respects_budget_order_lookup_and_original_inputs(
    texts: list[str], budget: int
) -> None:
    document_id = uuid4()
    hits = [
        make_hit(
            text, document_id=document_id, chunk_index=index, page_number=index + 1
        )
        for index, text in enumerate(texts)
    ]
    original = [hit.model_dump() for hit in hits]

    # Character counts are another deterministic stand-in. They are not an
    # estimate of model tokens; this exercises the supplied counter's contract.
    context = build_context(hits, max_tokens=budget, count_tokens=len)

    assert len(context.text) <= budget
    assert [hit.model_dump() for hit in hits] == original
    assert list(context.sources) == [
        f"S{rank}" for rank in range(1, len(context.sources) + 1)
    ]
    indices = [source.chunk_index for source in context.sources.values()]
    assert indices == sorted(set(indices))
    assert context.text == "\n\n".join(
        block_for(hits[source.chunk_index], label)
        for label, source in context.sources.items()
    )
    for label, source in context.sources.items():
        assert source.source_id == label
        assert source.document_id == document_id
        assert source.page_number == source.chunk_index + 1


@given(
    texts=st.lists(st.text(min_size=1, max_size=80), max_size=8),
    budget=st.integers(min_value=1, max_value=800),
)
def test_generated_selection_matches_greedy_whole_block_reference(
    texts: list[str], budget: int
) -> None:
    document_id = uuid4()
    hits = [
        make_hit(text, document_id=document_id, chunk_index=index)
        for index, text in enumerate(texts)
    ]
    expected = ""
    selected = []
    for index, hit in enumerate(hits):
        label = f"S{len(selected) + 1}"
        candidate = expected + ("\n\n" if selected else "") + block_for(hit, label)
        if len(candidate) <= budget:
            expected = candidate
            selected.append(index)

    context = build_context(hits, max_tokens=budget, count_tokens=len)

    assert context.text == expected
    assert [source.chunk_index for source in context.sources.values()] == selected
