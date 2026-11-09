import math
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MAX_CONTRACT_DEPTH = 64


def _validate_contract_values(value: Any) -> None:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > _MAX_CONTRACT_DEPTH:
            raise ValueError("request data is nested too deeply")
        if isinstance(current, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in current):
                raise ValueError("request strings must contain valid Unicode")
        elif isinstance(current, float) and not math.isfinite(current):
            raise ValueError("request numbers must be finite")
        elif isinstance(current, dict):
            pending.extend((key, depth + 1) for key in current)
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend((item, depth + 1) for item in current)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_request_values(cls, value: Any) -> Any:
        _validate_contract_values(value)
        return value


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
    pass


class Sampling(Contract):
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class ModelCallRequest(Contract):
    run_id: str = Field(min_length=1, max_length=256)
    step: str = Field(min_length=1, max_length=128)
    role: Literal["triager"]
    messages: list[dict[str, Any]] = Field(min_length=1)
    sampling: Sampling = Field(default_factory=Sampling)
    taint: int = Field(ge=0, le=3)
    source: Literal["github_issue", "user_feedback"]


class ModelOutcome(Contract):
    outcome: Literal[
        "completed", "quota_exhausted", "throttled", "failed", "ledger_unavailable"
    ]
    request_id: str
    decided_at: datetime
    model_pinned: str
    run_id: str
    step: str
    model_reported: str | None = None
    text: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retry_at: datetime | None = None
    retry_in: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    scope: (
        Literal[
            "minute_requests",
            "minute_tokens",
            "day_requests",
            "day_tokens",
            "week_requests",
        ]
        | None
    ) = None
    http_status: int | None = None
    error_code: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome_fields(self) -> Self:
        if self.outcome == "completed" and not (
            (self.text is not None and self.text.strip()) or self.tool_calls
        ):
            raise ValueError("completed outcomes require text or tool calls")
        if self.outcome == "quota_exhausted" and (
            self.retry_at is None or self.scope is None
        ):
            raise ValueError("quota_exhausted outcomes require retry_at and scope")
        if self.outcome == "throttled" and self.retry_in is None:
            raise ValueError("throttled outcomes require retry_in")
        if self.outcome in {"failed", "ledger_unavailable"} and (
            not self.error_code or not self.reason
        ):
            raise ValueError("failed outcomes require error_code and reason")
        return self


class IssueRunRequest(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=100_000)
    read_basis: datetime | None = None


class TriageResult(Contract):
    kind: Literal["bug", "feature", "question", "docs", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, max_length=300)


class IssueRunResult(Contract):
    issue_id: str
    round_id: str
    event_id: str
    run_id: str
    proposal_id: str
    status: Literal["committed", "quota_exhausted", "throttled"]
    model: str | None = None
    provider: Literal["groq"] | None = None
    retry_at: datetime | None = None
    retry_in: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    scope: (
        Literal[
            "minute_requests",
            "minute_tokens",
            "day_requests",
            "day_tokens",
            "week_requests",
        ]
        | None
    ) = None
    decided_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    triage: TriageResult | None = None
    triage_error: Literal["unparseable"] | None = None

    @model_validator(mode="after")
    def validate_triage_fields(self) -> Self:
        if self.triage is not None and self.triage_error is not None:
            raise ValueError("triage and triage_error cannot both be set")
        if self.status != "committed" and (
            self.triage is not None or self.triage_error is not None
        ):
            raise ValueError("only committed results carry triage fields")
        return self


class RunStartAccepted(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    state: Literal["submitted"] = "submitted"
    invocation_id: str | None = None


class RunStatus(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    state: Literal["running"] = "running"
