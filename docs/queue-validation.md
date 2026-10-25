# Queue milestone validation — 25 October 2026

The user verified real PDF upload, job success, and a working `/ask` response.
The implementation includes SQLite jobs, RQ execution, failure hooks, guarded
recovery, bounded retry chains, atomic same-ID redelivery, sweep, and timer setup.

Manual reindexing now acquires the shared document lock and rejects unfinished
jobs. Tests cover active ownership, lock lifetime through index publication, and
queued/running/failed readiness. RQ abandonment is left to guarded recovery,
because a dead parent does not prove its child stopped processing.

## Verification

- Full offline suite: 927 passed, three opt-in Redis tests skipped, two dependency
  deprecation warnings. Ruff lint/formatting and whitespace checks passed.
- Three real Redis integration tests previously passed: atomic uniqueness,
  concurrent sweeps, and successful enqueue with a lost acknowledgement.
- Real isolated restart/crash walkthrough passed:
  - A queued task survived broker container restart with AOF enabled.
  - Killing the parent left its child running and holding document ownership.
  - After real RQ abandonment/registry expiry, the restarted worker recorded
    failure in RQ while SQLite remained running for guarded recovery.
  - The sweep skipped the live orphan child, preserving its running record.
  - After terminating that child, recovery marked the original failed and
    created/enqueued attempt 2 for the same document.
  - The real cached MiniLM worker completed attempt 2 and saved two vectors.
  - The private container, volume, temporary records, and test child were cleaned
    up; the application broker and documents were not restarted or removed.
- No hosted provider calls or new model downloads were made for this closeout.
- Test Redis uses a private container, private volume, private queue, temporary
  SQLite/PDF/index paths, and existing cached CPU MiniLM.

## Deliberate performance deviation

The original plan asked for one cached embedding model per worker. The default
RQ worker forks one child per task; model loading currently happens in that child.
Cross-task model reuse is deferred so we retain task process isolation. A future
change must measure the overhead and validate the selected process model with
PyTorch, heartbeats, hard timeouts, crash recovery, and memory limits.

## Remaining manual handoff

- Restart API and worker to load the latest locking and hook behavior.
- The installed user timer was observed active and firing. Its live sweep reports
  four queued records with missing PDFs and exits nonzero. Those records were
  preserved: restore their original PDFs or explicitly review/cancel stale test
  records before expecting a clean timer summary. No user records were deleted.
- Validate answers/citations against the fictional handbook: annual leave is
  25 days per year on page 1; remote work is up to 3 days/week with manager
  approval on page 2. The handbook does not provide a maternity-leave policy.
  A successful HTTP response alone does not establish answer accuracy.
- Commit/push and confirm GitHub CI. No commit or push was performed here.
