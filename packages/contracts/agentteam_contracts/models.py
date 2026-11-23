import math
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MAX_CONTRACT_DEPTH = 64
ModelRole = Literal["triager", "reproducer", "fixer", "tester"]


def _is_safe_contract_path(path: str) -> bool:
    if not path or path != path.strip():
        return False
    if path.startswith(("/", "\\", "~")) or any(
        ch in path for ch in ("\x00", "\n", "\r", "\\")
    ):
        return False
    if any(part in ("", ".", "..", ".git") for part in path.split("/")):
        return False
    parsed = PurePosixPath(path)
    return not parsed.is_absolute() and ".." not in parsed.parts


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
    fact_ids_injected: list[str] = Field(default_factory=list)
    fact_ids_cited: list[str] = Field(default_factory=list)
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
    fact_ids_injected: list[str] = Field(default_factory=list)
    fact_ids_cited: list[str] = Field(default_factory=list)
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


class RepoPush(Contract):
    branch: Literal["main"] = "main"
    before: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    after: str = Field(pattern=r"^[0-9a-f]{40}$")


class RepoPushResponse(Contract):
    processed: list[str]
    already_recorded: list[str]
    closed_fact_ids: list[str]
    main_head: str


class SymbolLocation(Contract):
    file: str
    line: int = Field(ge=1)

    @field_validator("file")
    @classmethod
    def safe_path(cls, value: str) -> str:
        if not _is_safe_contract_path(value):
            raise ValueError("file must be a safe repository-relative path")
        return value


class SymbolsResolve(Contract):
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    names: list[str] = Field(default_factory=list, max_length=200)
    locations: list[SymbolLocation] = Field(default_factory=list, max_length=200)

    @field_validator("names")
    @classmethod
    def valid_names(cls, values: list[str]) -> list[str]:
        if any("\x00" in value for value in values):
            raise ValueError("names must not contain NUL")
        return values


class SymbolsResolveResponse(Contract):
    symbols: list[str]


class Sampling(Contract):
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class ModelCallRequest(Contract):
    run_id: str = Field(min_length=1, max_length=256)
    step: str = Field(min_length=1, max_length=128)
    role: ModelRole
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = Field(default=None, min_length=1)
    tool_choice: Literal["auto", "required"] | None = None
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
    provider_error_body: str | None = None

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
    repo: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=100_000)
    read_basis: datetime | None = None
    attempt: int = Field(default=1, ge=1, le=99)


class TriageResult(Contract):
    kind: Literal["bug", "feature", "question", "docs", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, max_length=300)
    area: str


class IssueRunResult(Contract):
    issue_id: str
    round_id: str
    event_id: str
    run_id: str
    proposal_id: str
    status: Literal["committed", "quota_exhausted", "throttled"]
    attempt: int = Field(default=1, ge=1, le=99)
    model: str | None = None
    provider: str | None = None
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
    area: str = "unknown"
    triage: TriageResult | None = None
    triage_error: Literal["unparseable"] | None = None

    @field_validator("triage", mode="before")
    @classmethod
    def default_missing_triage_area(cls, value: Any) -> Any:
        # Results stored before triage.area existed omit the field.
        if isinstance(value, dict) and "area" not in value:
            return {**value, "area": "unknown"}
        return value

    @model_validator(mode="after")
    def validate_triage_fields(self) -> Self:
        if self.triage is not None and self.triage_error is not None:
            raise ValueError("triage and triage_error cannot both be set")
        if self.triage is not None and self.area != self.triage.area:
            raise ValueError("result area must match triage area")
        if self.status != "committed" and (
            self.triage is not None or self.triage_error is not None
        ):
            raise ValueError("only committed results carry triage fields")
        return self


class RunStartAccepted(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    state: Literal["submitted"] = "submitted"
    invocation_id: str | None = None
    attempt: int = Field(default=1, ge=1, le=99)


class RunStatus(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    state: Literal["running"] = "running"
    attempt: int = Field(default=1, ge=1, le=99)


class IssueRunRequestV2(IssueRunRequest):
    base_commit: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{7,64}$")
    fact_condition: Literal["C0", "C2"] | None = None
    fact_snapshot: list[dict[str, Any]] | None = Field(default=None, max_length=500)
    pr_branch_suffix: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9._-]{1,20}$"
    )

    @model_validator(mode="after")
    def validate_fact_condition(self) -> Self:
        if self.fact_condition == "C2" and self.fact_snapshot is None:
            raise ValueError("C2 requires a frozen fact snapshot")
        if self.fact_condition == "C0" and self.fact_snapshot is not None:
            raise ValueError("C0 cannot carry a fact snapshot")
        return self


class RepoHeadResponse(Contract):
    repo: str = Field(min_length=1, max_length=200)
    default_branch: str = Field(min_length=1, max_length=255)
    head_sha: str = Field(pattern=r"^[0-9a-fA-F]{7,64}$")


class PrFileChange(Contract):
    path: str = Field(min_length=1, max_length=500)
    content: str | None = None

    @field_validator("path")
    @classmethod
    def validate_safe_path(cls, value: str) -> str:
        if not _is_safe_contract_path(value):
            raise ValueError("file path must be a safe relative repository path")
        return value


class OpenPrRequest(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    issue_id: str = Field(min_length=1, max_length=200)
    attempt: int = Field(default=1, ge=1, le=99)
    base_commit: str = Field(pattern=r"^[0-9a-fA-F]{7,64}$")
    base_branch: str | None = Field(default=None, min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(default="", max_length=60_000)
    files: list[PrFileChange] = Field(min_length=1, max_length=100)
    branch_suffix: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,20}$")


class EvidenceFactItem(Contract):
    fact_id: str = Field(min_length=1, max_length=128)
    claim: str = Field(min_length=1, max_length=500)
    checked_at_sha: str = Field(pattern=r"^[0-9a-fA-F]{7,64}$")
    valid_at_base: bool
    taint: int = Field(ge=0, le=3)


class SyncDecisionItem(Contract):
    step: Literal["triage", "repro", "fix", "test"]
    round_id: str = Field(min_length=1, max_length=256)
    proposal_id: str = Field(min_length=1, max_length=256)
    op: str = Field(min_length=1, max_length=64)
    status: Literal["accepted", "rejected"]
    reason: str = Field(default="", max_length=500)


class ModelUsageItem(Contract):
    role: ModelRole
    model: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=100)


class EvidenceCommentRequest(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    issue_id: str = Field(min_length=1, max_length=200)
    attempt: int = Field(default=1, ge=1, le=99)
    base_commit: str = Field(pattern=r"^[0-9a-fA-F]{7,64}$")
    head_sha: str = Field(pattern=r"^[0-9a-fA-F]{7,64}$")
    triage: TriageResult | None = None
    repro_test_node_id: str = Field(min_length=1, max_length=500)
    repro_failed_before: bool
    repro_passed_after: bool
    suite_passed: bool
    suite_summary: str = Field(default="", max_length=500)
    review_verdict: Literal["approve", "changes_requested"]
    review_summary: str = Field(default="", max_length=500)
    repro_files: list[str] = Field(default_factory=list)
    fix_files: list[str] = Field(default_factory=list)
    sync_decisions: list[SyncDecisionItem] = Field(default_factory=list)
    cited_facts: list[EvidenceFactItem] = Field(default_factory=list)
    fact_ids_injected: list[str] = Field(default_factory=list)
    fact_ids_cited: list[str] = Field(default_factory=list)
    models_used: list[ModelUsageItem] = Field(default_factory=list)


class CommitStatusRequest(Contract):
    idempotency_key: str = Field(min_length=1, max_length=256)
    state: Literal["pending", "success", "failure", "error"]
    description: str = Field(min_length=1, max_length=140)
    context: str = Field(default="agent-team", min_length=1, max_length=100)


class GithubWriteOutcome(Contract):
    outcome: Literal["ok", "throttled", "failed"]
    decided_at: datetime
    retry_in: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retry_at: datetime | None = None
    error_code: str | None = None
    reason: str | None = None
    pr_number: int | None = Field(default=None, ge=1)
    html_url: str | None = None
    head_sha: str | None = None
    branch: str | None = None
    comment_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_write_outcome(self) -> Self:
        if self.outcome == "ok":
            if any(
                value is not None
                for value in (
                    self.retry_in,
                    self.retry_at,
                    self.error_code,
                    self.reason,
                )
            ):
                raise ValueError("ok outcomes cannot carry retry or error fields")
        elif self.outcome == "throttled":
            if self.retry_in is None:
                raise ValueError("throttled outcomes require retry_in")
        elif not self.error_code or not self.reason:
            raise ValueError("failed outcomes require error_code and reason")
        return self


class ApprovalDecision(Contract):
    approved: bool
    by: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=2000)
    head_sha: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{7,64}$")
    review_id: int | None = Field(default=None, ge=1)
    source: Literal["github_review", "cli"] = "github_review"


class ApprovalRequest(ApprovalDecision):
    issue_id: str = Field(min_length=1, max_length=200)
    attempt: int = Field(default=1, ge=1, le=99)


class PrHeadUpdateRequest(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    attempt: int = Field(default=1, ge=1, le=99)
    pr_number: int = Field(ge=1)
    head_sha: str = Field(pattern=r"^[0-9a-fA-F]{7,64}$")


class ApprovalResult(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    attempt: int = Field(default=1, ge=1, le=99)
    status: Literal["recorded", "already_decided", "stale_head"]
    stage: (
        Literal["running", "awaiting_approval", "merged", "closed", "failed"] | None
    ) = None


class RunStatusV2(Contract):
    issue_id: str = Field(min_length=1, max_length=200)
    state: Literal["running", "awaiting_approval"] = "running"
    attempt: int = Field(default=1, ge=1, le=99)
    pr_number: int | None = Field(default=None, ge=1)
    pr_url: str | None = None
    head_sha: str | None = None


class IssueRunResultV2(Contract):
    issue_id: str
    attempt: int = Field(default=1, ge=1, le=99)
    status: Literal[
        "approved",
        "rejected",
        "failed",
        "quota_exhausted",
        "throttled",
    ]
    stage: Literal["merged", "closed", "failed"]
    base_commit: str | None = None
    area: str = "unknown"
    triage: TriageResult | None = None
    triage_error: Literal["unparseable"] | None = None
    repro_test_node_id: str | None = None
    repro_files: list[str] = Field(default_factory=list)
    fix_files: list[str] = Field(default_factory=list)
    fix_attempts: int = Field(default=0, ge=0, le=2)
    pr_number: int | None = Field(default=None, ge=1)
    pr_url: str | None = None
    pr_head_sha: str | None = None
    approval: ApprovalDecision | None = None
    failure_reason: str | None = None
    model: str | None = None
    provider: str | None = None
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


class FactQuery(Contract):
    issue_refs: list[str] = Field(default_factory=list, max_length=100)
    symbols: list[str] = Field(default_factory=list)
    base_commit: str = Field(min_length=1, max_length=64)
    ancestry_filter: bool = True
    limit: int = Field(default=50, ge=1, le=1000)

    @field_validator("issue_refs")
    @classmethod
    def validate_issue_refs(cls, values: list[str]) -> list[str]:
        if any(
            re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*", value) is None
            for value in values
        ):
            raise ValueError("issue references must use owner/repo#number")
        return list(dict.fromkeys(value.lower() for value in values))

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, symbols: list[str]) -> list[str]:
        if any("\x00" in s for s in symbols):
            raise ValueError("symbol names must not contain NUL characters")
        return symbols


class FactView(Contract):
    id: str
    claim: str = ""
    kind: str = ""
    area: str = ""
    anchors: list[dict[str, Any]] = Field(default_factory=list)
    valid_from_commit: str
    closing_commits: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_half_life_days: float = Field(default=90.0, gt=0.0)
    last_confirmed_at: datetime | None = None
    taint: int = Field(default=0, ge=0, le=3)
    evidence_test: str | None = None
    source_issue: str | None = None
    supersedes: str | None = None


class FactQueryResponse(Contract):
    base_commit: str
    facts: list[FactView] = Field(default_factory=list)
    dropped: dict[str, int] = Field(default_factory=dict)


class CloseByIssueRequest(Contract):
    issue_ref: str = Field(min_length=1, max_length=200)
    closing_commit: str = Field(min_length=1, max_length=64)
    reason: Literal["linked_fix", "issue_closed", "issue_closed_not_planned"]


class CloseByIssueResponse(Contract):
    closed_fact_ids: list[str] = Field(default_factory=list)
