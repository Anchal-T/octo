from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventWrite(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    source: Literal[
        "user_feedback", "github_issue", "telemetry", "maintainer", "agent", "tool"
    ]
    kind: str = Field(min_length=1, max_length=64)
    taint: int = Field(ge=0, le=3)
    payload: dict[str, Any]


class EventRecord(Contract):
    id: str
    source: str
    kind: str
    taint: int
    taint_label: str
    payload: dict[str, Any]


class RoundWrite(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    seq: int = Field(ge=1)
    base_commit: str | None = None
    read_basis: datetime


class RoundRecord(Contract):
    id: str
    seq: int
    status: Literal["open", "syncing", "committed"]
    base_commit: str | None
    read_basis: datetime
    committed_proposals: list[str]


class RunWrite(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    agent_id: str
    role: str
    round_id: str
    workflow_id: str
    step: str
    model: str | None = None
    base_commit: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    outcome: str
    event_ids: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None


class RunRecord(Contract):
    id: str
    agent_id: str
    role: str
    round_id: str
    workflow_id: str
    step: str
    model: str | None
    base_commit: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    outcome: str
    event_ids: list[str]
    started_at: datetime
    ended_at: datetime | None


class ProposalWrite(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    round_id: str
    agent_id: str
    run_id: str
    op: str
    evidence_event_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any]


class ProposalRecord(Contract):
    id: str
    round_id: str
    agent_id: str
    run_id: str
    op: str
    status: Literal["pending", "accepted", "rejected"]
    reason: str | None
    evidence_event_ids: list[str]
    payload: dict[str, Any]


class CommitRound(Contract):
    proposal_ids: list[str] = Field(min_length=1)


class IssueRunRequest(Contract):
    issue_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=100_000)
    read_basis: datetime | None = None


class IssueRunResult(Contract):
    issue_id: str
    round_id: str
    event_id: str
    run_id: str
    proposal_id: str
    status: Literal["committed"]
