import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
import restate
from agentteam_contracts import (
    EventWrite,
    IssueRunRequest,
    IssueRunResult,
    ModelOutcome,
)

from services.workflow.app import main as workflow
from services.workflow.app.plan import (
    build_issue_plan_v1,
    build_triage_proposal_v1,
    map_triage_return_v1,
    map_triage_run_v1,
)
from services.workflow.app.retries import decide_wait


def test_v1_plan_uses_separate_memory_keys_and_maps_model_completion():
    basis = datetime(2026, 10, 31, 12, tzinfo=UTC)
    request = IssueRunRequest(
        issue_id="issue-17",
        title="Example title",
        body="Example body",
        read_basis=basis,
    )
    outcome = ModelOutcome(
        outcome="completed",
        request_id="mc-example",
        decided_at=basis,
        model_pinned="openai/gpt-oss-20b",
        model_reported="openai/gpt-oss-20b",
        run_id="run-example",
        step="triage",
        text="Private triage result",
        tokens_in=12,
        tokens_out=3,
        started_at=basis,
        finished_at=basis,
    )

    plan = build_issue_plan_v1(request)
    run = map_triage_run_v1(plan, outcome)
    proposal = build_triage_proposal_v1(plan, outcome)

    assert plan.round.idempotency_key == "issue-17:v1:round:1"
    assert plan.event.idempotency_key == "issue-17:v1:issue"
    assert run.idempotency_key == "issue-17:v1:triage:1"
    assert proposal.idempotency_key == "issue-17:v1:triage-proposal:1"
    assert run.model == "openai/gpt-oss-20b"
    assert run.outcome == "completed"
    assert run.tokens_in == 12
    assert run.tokens_out == 3
    assert proposal.payload == {
        "stub": True,
        "issue_id": "issue-17",
        "routing": "not_implemented",
        "model": "openai/gpt-oss-20b",
        "request_id": "mc-example",
    }
    assert "Private triage result" not in str(proposal.payload)


def test_decide_wait_includes_exact_two_hour_cap():
    decided_at = datetime(2026, 10, 31, 12, tzinfo=UTC)

    assert decide_wait(
        retry_at=decided_at + timedelta(hours=2),
        decided_at=decided_at,
        cap_s=7200,
    ) == {"action": "sleep", "wait_s": 7200}
    assert decide_wait(
        retry_at=decided_at + timedelta(hours=2, seconds=1),
        decided_at=decided_at,
        cap_s=7200,
    ) == {"action": "return", "wait_s": None}


def test_issue_run_result_adds_throttle_metadata_without_changing_old_fields():
    decided_at = datetime(2026, 10, 31, 12, tzinfo=UTC)
    result = IssueRunResult(
        issue_id="issue-17",
        round_id="round-17",
        event_id="event-17",
        run_id="run-17",
        proposal_id="proposal-17",
        status="throttled",
        model="openai/gpt-oss-20b",
        provider="groq",
        retry_in=30,
        decided_at=decided_at,
        started_at=decided_at,
        finished_at=decided_at,
    )

    assert result.status == "throttled"
    assert result.model == "openai/gpt-oss-20b"
    assert result.provider == "groq"
    assert result.retry_in == 30
    assert result.decided_at == decided_at
    assert result.model_dump(mode="json")["status"] == "throttled"


def test_committed_issue_result_keeps_legacy_required_fields():
    result = IssueRunResult(
        issue_id="issue-17",
        round_id="round-17",
        event_id="event-17",
        run_id="run-17",
        proposal_id="proposal-17",
        status="committed",
    )

    assert result.status == "committed"
    assert result.model is None
    assert result.provider is None
    assert result.retry_in is None


def test_throttle_wait_over_two_hours_returns_without_sleeping():
    basis = datetime(2026, 10, 31, 12, tzinfo=UTC)
    model_outcome = ModelOutcome(
        outcome="throttled",
        request_id="mc-example",
        decided_at=basis,
        model_pinned="openai/gpt-oss-20b",
        run_id="run-example",
        step="triage",
        retry_in=31_536_000,
        started_at=basis,
        finished_at=basis,
    )

    class Context:
        def __init__(self):
            self.sleeps = []

        async def run_typed(self, name, *args, **kwargs):
            if name == "read-basis":
                return basis.isoformat()
            if name in ("triage-call:1", "triage-call:2"):
                return model_outcome
            if name == "triage-throttle-wait:1":
                return args[0](**kwargs)
            if name in ("open-round", "record-issue"):
                return {}
            raise AssertionError(f"unexpected workflow step: {name}")

        async def sleep(self, duration, *, name):
            self.sleeps.append((duration, name))

    context = Context()
    request = IssueRunRequest(issue_id="issue-17", title="Example title")

    result = asyncio.run(
        workflow.run_issue_v1(cast(restate.WorkflowContext, context), request)
    )

    assert result.status == "throttled"
    assert result.retry_in == 31_536_000
    assert context.sleeps == []


def test_issue_plan_validation_error_is_terminal(monkeypatch):
    basis = datetime(2026, 10, 31, 12, tzinfo=UTC)
    request = IssueRunRequest(
        issue_id="issue-17", title="Example title", read_basis=basis
    )

    class Context:
        async def run_typed(self, *args, **kwargs):
            raise AssertionError("no durable steps should run after an invalid plan")

    monkeypatch.setattr(
        workflow,
        "build_issue_plan_v1",
        lambda request: EventWrite(idempotency_key="invalid"),
    )

    with pytest.raises(restate.TerminalError, match="invalid triager issue plan"):
        asyncio.run(
            workflow.run_issue_v1(cast(restate.WorkflowContext, Context()), request)
        )


def test_triage_plan_validation_error_is_terminal_after_model_call(monkeypatch):
    basis = datetime(2026, 10, 31, 12, tzinfo=UTC)
    request = IssueRunRequest(
        issue_id="issue-17", title="Example title", read_basis=basis
    )
    plan = workflow.build_issue_plan_v1(request)
    model_outcome = ModelOutcome(
        outcome="completed",
        request_id="mc-example",
        decided_at=basis,
        model_pinned="openai/gpt-oss-20b",
        run_id=plan.run_id,
        step="triage",
        text="Example triage",
        started_at=basis,
        finished_at=basis,
    )

    class Context:
        async def run_typed(self, name, *args, **kwargs):
            if name == "triage-call:1":
                return model_outcome
            return {}

    def invalid_proposal(*args):
        return EventWrite(idempotency_key="invalid")

    monkeypatch.setattr(workflow, "build_triage_proposal_v1", invalid_proposal)

    with pytest.raises(restate.TerminalError, match="triage result"):
        asyncio.run(
            workflow.run_issue_v1(cast(restate.WorkflowContext, Context()), request)
        )


def test_second_throttle_result_preserves_retry_and_model_metadata():
    basis = datetime(2026, 10, 31, 12, tzinfo=UTC)
    plan = build_issue_plan_v1(
        IssueRunRequest(
            issue_id="issue-17",
            title="Example title",
            read_basis=basis,
        )
    )
    outcome = ModelOutcome(
        outcome="throttled",
        request_id="mc-example",
        decided_at=basis,
        model_pinned="openai/gpt-oss-20b",
        run_id=plan.run_id,
        step="triage",
        retry_count=1,
        retry_in=30,
        started_at=basis,
        finished_at=basis,
    )

    result = map_triage_return_v1(plan, outcome)

    assert result.status == "throttled"
    assert result.retry_in == 30
    assert result.model == "openai/gpt-oss-20b"
    assert result.provider == "groq"
    assert result.started_at == basis
    assert result.finished_at == basis
    assert result.decided_at == basis
