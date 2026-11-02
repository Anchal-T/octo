from datetime import UTC, datetime

import pytest
from agentteam_contracts import IssueRunRequest
from pydantic import ValidationError

from services.workflow.app.plan import build_issue_plan, build_issue_plan_v1


def test_issue_request_leaves_read_basis_stable_for_retries():
    request = IssueRunRequest(issue_id="issue-17", title="Example title")

    assert request.read_basis is None
    with pytest.raises(ValueError, match="durable workflow"):
        build_issue_plan(request)


def test_issue_id_limit_keeps_v1_derived_keys_within_contract_limit():
    basis = datetime(2026, 10, 26, tzinfo=UTC)
    request = IssueRunRequest(
        issue_id="i" * 200,
        title="Example title",
        read_basis=basis,
    )

    plan = build_issue_plan_v1(request)

    assert (
        max(
            len(plan.round.idempotency_key),
            len(plan.event.idempotency_key),
            len(plan.run_key),
            len(plan.proposal_key),
        )
        <= 256
    )
    with pytest.raises(ValidationError):
        IssueRunRequest(issue_id="i" * 201, title="Example title")


def test_issue_plan_is_repeatable_and_keeps_user_text_untrusted():
    request = IssueRunRequest(
        issue_id="issue-17",
        title="Example title",
        body="Example body",
        read_basis=datetime(2026, 10, 26, tzinfo=UTC),
    )

    first = build_issue_plan(request)
    replay = build_issue_plan(request)

    assert first.round_id == replay.round_id
    assert first.event_id == replay.event_id
    assert first.run_id == replay.run_id
    assert first.proposal_id == replay.proposal_id
    assert first.event.taint == 2
    assert first.event.payload["body"] == "Example body"
    assert first.proposal.payload["stub"] is True
