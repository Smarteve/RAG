# Retrieval evaluation baseline — 30 October 2026

The bundled `employee-handbook-v1` case set contains seven answerable questions
with expected source phrases/pages and one unsupported parking question.
Positive labels were checked against the actual extracted sample PDF.

Offline validation after adding the evaluator: 1,051 tests passed, three opt-in
Redis checks skipped. Ruff lint, formatting, README command syntax, and whitespace
checks passed. Unit coverage includes scoring/rank cutoffs, metric denominators,
ground-truth validation, source mismatches, stale indexes, deleted/unfinished
documents, document ownership, backend failures, and CLI report output.

Source: `examples/employee-handbook.pdf`, SHA-256
`fa98cd8876120e82840d396a8060e3d91ed6eec1f5451ec41617ee0fe58fbd69`.
The model was the existing cached CPU
`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Recorded real-model check

Both versions were prepared from the same PDF with the same model, character
cap of 1000, requested character overlap of 200, questions, and top-k of 5.
Each version used its own document ID and private in-memory Qdrant collection.
Artifacts and SQLite records were isolated in temporary storage and removed
afterward. The application collections, document/job records, Redis, and Docker
services were not modified. Downloads were disabled; no hosted generation was
called.

| Chunking | Chunks | Answerable cases | Unsupported cases | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Character windows | 2 | 7 | 1 | 1.0 | 1.0 |
| Token-aware windows | 2 | 7 | 1 | 1.0 | 1.0 |

All seven answerable questions had their first relevant hit at rank 1 in both
runs. The unsupported question retained retrieved neighbours but received no
relevance rank and was excluded from the metrics.

Each run was repeated against the exact local cosine reference using the same
saved vectors and questions. Returned chunk identities/orders matched for all
eight queries; the largest absolute score difference was approximately
`2.30e-8`, within the checked `1e-6` tolerance. This is actual-model parity for
this sample, beyond the existing synthetic-vector tests.

## What these results establish

The evaluator works with the cached tokenizer/model, real vector adapter, saved
published generations, source labels, and the local cosine reference. Both
chunking methods produce the same two chunks on this short handbook; these
results show no retrieval-quality improvement from token-aware splitting.

With only two chunks, Hit@5 is an easy ceiling. Harder/larger evaluation documents
are needed to compare chunking quality meaningfully. Exact phrase/page matching
can miss semantically equivalent support and treats a phrase split across chunks
as a miss. The labels and scoring policy are deliberately explicit and editable.

This check used Qdrant's in-memory implementation, not the Docker server. It
does not establish server persistence/restart behavior, answer quality,
unsupported-answer refusal, or generation citation correctness. Run the README's
evaluation command against a completed Docker upload for server evidence; keep
hosted answer evaluation manual.
