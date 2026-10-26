from dataclasses import dataclass

from agentteam_common import stable_id
from agentteam_contracts import (
    EventWrite,
    IssueRunRequest,
    ProposalWrite,
    RoundWrite,
    RunWrite,
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
