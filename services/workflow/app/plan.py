from dataclasses import dataclass

from agentteam_common import stable_id
from agentteam_contracts import (
    EventWrite,
    IssueRunRequest,
    IssueRunResult,
    ModelOutcome,
    ProposalWrite,
    RoundWrite,
    RunWrite,
    TriageResult,
)


@dataclass(frozen=True)
class IssuePlan:
    round: RoundWrite
    event: EventWrite
    run: RunWrite
    proposal: ProposalWrite
    round_id: str
    event_id: str
    run_id: str
    proposal_id: str


@dataclass(frozen=True)
class IssuePlanV1:
    round: RoundWrite
    event: EventWrite
    issue_id: str
    round_id: str
    event_id: str
    run_id: str
    proposal_id: str
    run_key: str
    proposal_key: str


def build_issue_plan(request: IssueRunRequest) -> IssuePlan:
    if request.read_basis is None:
        raise ValueError("read_basis must be resolved in the durable workflow first")

    round_key = f"{request.issue_id}:round:1"
    event_key = f"{request.issue_id}:issue"
    run_key = f"{request.issue_id}:triage:1"
    proposal_key = f"{request.issue_id}:triage-proposal:1"

    round_id = stable_id("round", round_key)
    event_id = stable_id("ev", event_key)
    run_id = stable_id("run", run_key)
    proposal_id = stable_id("prop", proposal_key)

    return IssuePlan(
        round=RoundWrite(
            idempotency_key=round_key,
            seq=1,
            base_commit=None,
            read_basis=request.read_basis,
        ),
        event=EventWrite(
            idempotency_key=event_key,
            source="github_issue",
            kind="issue",
            taint=2,
            payload={
                "issue_id": request.issue_id,
                "title": request.title,
                "body": request.body,
            },
        ),
        run=RunWrite(
            idempotency_key=run_key,
            agent_id="stub-triager",
            role="triager",
            round_id=round_id,
            workflow_id=request.issue_id,
            step="triage",
            outcome="stub_completed",
            event_ids=[event_id],
            started_at=request.read_basis,
            ended_at=request.read_basis,
        ),
        proposal=ProposalWrite(
            idempotency_key=proposal_key,
            round_id=round_id,
            agent_id="stub-triager",
            run_id=run_id,
            op="route_issue",
            evidence_event_ids=[event_id],
            payload={
                "stub": True,
                "issue_id": request.issue_id,
                "routing": "not_implemented",
            },
        ),
        round_id=round_id,
        event_id=event_id,
        run_id=run_id,
        proposal_id=proposal_id,
    )


def build_issue_plan_v1(request: IssueRunRequest) -> IssuePlanV1:
    if request.read_basis is None:
        raise ValueError("read_basis must be resolved in the durable workflow first")

    round_key = f"{request.issue_id}:v1:round:1"
    event_key = f"{request.issue_id}:v1:issue"
    run_key = f"{request.issue_id}:v1:triage:1"
    proposal_key = f"{request.issue_id}:v1:triage-proposal:1"

    round_id = stable_id("round", round_key)
    event_id = stable_id("ev", event_key)
    run_id = stable_id("run", run_key)
    proposal_id = stable_id("prop", proposal_key)

    return IssuePlanV1(
        round=RoundWrite(
            idempotency_key=round_key,
            seq=1,
            base_commit=None,
            read_basis=request.read_basis,
        ),
        event=EventWrite(
            idempotency_key=event_key,
            source="github_issue",
            kind="issue",
            taint=2,
            payload={
                "issue_id": request.issue_id,
                "title": request.title,
                "body": request.body,
            },
        ),
        issue_id=request.issue_id,
        round_id=round_id,
        event_id=event_id,
        run_id=run_id,
        proposal_id=proposal_id,
        run_key=run_key,
        proposal_key=proposal_key,
    )


def map_triage_run_v1(plan: IssuePlanV1, outcome: ModelOutcome) -> RunWrite:
    if outcome.outcome != "completed":
        raise ValueError("only completed triage outcomes can be written")
    return RunWrite(
        idempotency_key=plan.run_key,
        agent_id="triager",
        role="triager",
        round_id=plan.round_id,
        workflow_id=plan.issue_id,
        step="triage",
        model=outcome.model_reported or outcome.model_pinned,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
        outcome="completed",
        event_ids=[plan.event_id],
        started_at=outcome.started_at or outcome.decided_at,
        ended_at=outcome.finished_at or outcome.decided_at,
    )


def map_triage_return_v1(plan: IssuePlanV1, outcome: ModelOutcome) -> IssueRunResult:
    if outcome.outcome not in ("quota_exhausted", "throttled"):
        raise ValueError("only retryable triage outcomes can be returned")
    return IssueRunResult(
        issue_id=plan.issue_id,
        round_id=plan.round_id,
        event_id=plan.event_id,
        run_id=plan.run_id,
        proposal_id=plan.proposal_id,
        status=outcome.outcome,
        model=outcome.model_reported or outcome.model_pinned,
        provider="groq",
        retry_at=outcome.retry_at,
        retry_in=outcome.retry_in,
        scope=outcome.scope,
        decided_at=outcome.decided_at,
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
    )


def map_triage_commit_v1(
    plan: IssuePlanV1, outcome: ModelOutcome, triage: TriageResult | None
) -> IssueRunResult:
    if outcome.outcome != "completed":
        raise ValueError("only completed triage outcomes can be committed")
    return IssueRunResult(
        issue_id=plan.issue_id,
        round_id=plan.round_id,
        event_id=plan.event_id,
        run_id=plan.run_id,
        proposal_id=plan.proposal_id,
        status="committed",
        model=outcome.model_reported or outcome.model_pinned,
        provider="groq",
        decided_at=outcome.decided_at,
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
        triage=triage,
        triage_error=None if triage is not None else "unparseable",
    )


def build_triage_proposal_v1(
    plan: IssuePlanV1, outcome: ModelOutcome, triage: TriageResult | None
) -> ProposalWrite:
    if outcome.outcome != "completed":
        raise ValueError("only completed triage outcomes can be proposed")
    return ProposalWrite(
        idempotency_key=plan.proposal_key,
        round_id=plan.round_id,
        agent_id="triager",
        run_id=plan.run_id,
        op="route_issue",
        evidence_event_ids=[plan.event_id],
        payload={
            "issue_id": plan.issue_id,
            "triage": None if triage is None else triage.model_dump(mode="json"),
            "triage_error": None if triage is not None else "unparseable",
            "model": outcome.model_reported or outcome.model_pinned,
            "request_id": outcome.request_id,
        },
    )
