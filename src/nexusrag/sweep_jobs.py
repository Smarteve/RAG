"""Run one recovery/retry pass; external scheduling remains separate."""

import argparse
import asyncio
import json
import sqlite3

from nexusrag import config
from nexusrag.services.delivery_reconciliation import reconcile_queued_jobs
from nexusrag.services.job_retry import (
    RetryAlreadyExistsError,
    RetryLimitReachedError,
    RetryNotAllowedError,
    RetrySubmissionError,
    load_retry_candidates,
    retry_failed_job,
)
from nexusrag.services.processing_lock import DocumentBusyError
from nexusrag.services.reconciliation import inspect_running_jobs


def _report(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


async def sweep_once(apply: bool = False, max_attempts: int = 3) -> int:
    """Recover, then retry eligible failed latest attempts. Return an exit code.

    Dry runs never create retries, enqueue tasks, or modify database records.
    Operational failures produce safe output and a nonzero exit code. Existing
    queued attempts are redelivered with their original ID only when RQ is absent.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    failures = 0
    retried = 0
    proposed = 0
    skipped = 0
    try:
        decisions = await inspect_running_jobs(
            config.JOB_DATABASE_PATH,
            config.REDIS_URL,
            apply_recovery=apply,
        )
    except sqlite3.Error, OSError, ValueError:
        _report("inspection_failed", reason="Could not inspect job records")
        return 1
    for decision in decisions:
        _report("inspection", **decision.model_dump(mode="json"))

    try:
        deliveries = await reconcile_queued_jobs(
            config.JOB_DATABASE_PATH, config.REDIS_URL, apply=apply
        )
    except sqlite3.Error, OSError, ValueError:
        _report(
            "delivery_inspection_failed", reason="Could not inspect queued attempts"
        )
        return 1
    for delivery in deliveries:
        _report("delivery", **delivery.model_dump(mode="json"))
        if delivery.action in {"uncertain", "delivery_unconfirmed"}:
            failures += 1

    try:
        # Query again after recovery so this includes newly recovered jobs and
        # jobs that were already marked failed by RQ hooks in earlier passes.
        candidates = await load_retry_candidates(config.JOB_DATABASE_PATH, max_attempts)
    except sqlite3.Error, OSError, ValueError:
        _report("candidate_lookup_failed", reason="Could not load retry candidates")
        return 1

    for job in candidates:
        if not apply:
            proposed += 1
            _report(
                "would_retry",
                job_id=str(job.job_id),
                attempt_number=job.attempt_number + 1,
            )
            continue
        try:
            retry = await retry_failed_job(
                job.job_id, config.JOB_DATABASE_PATH, max_attempts
            )
        except RetrySubmissionError as error:
            failures += 1
            _report(
                "retry_delivery_unconfirmed",
                job_id=str(error.job_id),
                reason="Queued attempt retained; delivery reconciliation required",
            )
        except RetryAlreadyExistsError as error:
            skipped += 1
            _report(
                "retry_skipped",
                job_id=str(job.job_id),
                current_job_id=str(error.job_id),
                reason="Later attempt already exists",
            )
        except DocumentBusyError, RetryLimitReachedError, RetryNotAllowedError:
            skipped += 1
            _report(
                "retry_skipped",
                job_id=str(job.job_id),
                reason="Document busy or retry no longer eligible",
            )
        except sqlite3.Error, OSError, ValueError:
            failures += 1
            _report(
                "retry_failed",
                job_id=str(job.job_id),
                reason="Could not persist retry or access source document",
            )
        else:
            retried += 1
            _report(
                "retry_enqueued",
                job_id=str(retry.job_id),
                previous_job_id=str(job.job_id),
                attempt_number=retry.attempt_number,
            )
    _report(
        "summary",
        apply=apply,
        recovered=sum(d.recovery_action == "failed" for d in decisions),
        redelivered=sum(d.action == "enqueued" for d in deliveries),
        proposed_deliveries=sum(d.action == "would_enqueue" for d in deliveries),
        retried=retried,
        proposed=proposed,
        skipped=skipped,
        failures=failures,
    )
    return 1 if failures else 0


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply recovery and enqueue eligible retries (default: dry run)",
    )
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=3,
        help="Total attempts per chain, including the original (default: 3)",
    )
    args = parser.parse_args()
    return asyncio.run(sweep_once(apply=args.apply, max_attempts=args.max_attempts))


if __name__ == "__main__":
    raise SystemExit(main())
