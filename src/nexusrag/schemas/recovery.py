"""Inspection decisions and optional guarded-recovery outcomes."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, StringConstraints

RecoveryDecisionStatus = Literal["active", "interrupted", "uncertain"]


class RecoveryDecision(BaseModel):
    """An observation plus an explicit recovery action, if requested.

    Callers must supply fixed, public-safe reasons rather than raw exceptions.
    """

    recovery_action: Literal["none", "skipped", "failed"] = "none"
    job_id: UUID
    decision: RecoveryDecisionStatus
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DeliveryDecision(BaseModel):
    """A queued-attempt delivery observation; no new processing attempt is made."""

    job_id: UUID
    action: Literal[
        "present",
        "would_enqueue",
        "enqueued",
        "skipped",
        "uncertain",
        "delivery_unconfirmed",
    ]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
