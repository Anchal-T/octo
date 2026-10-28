from datetime import UTC, datetime

from services.memory.app.scip import ScipDefinition, ScipIndex
from services.memory.app.sync import validate_proposal


def test_supported_triage_route_with_existing_evidence_is_accepted():
    proposal = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "run_id": "run-1",
        "op": "route_issue",
        "evidence_event_ids": ["ev-1"],
    }
    run = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "role": "triager",
        "event_ids": ["ev-1"],
    }
    evidence = {"ev-1": {"taint": 2}}

    assert validate_proposal("round-1", proposal, run, evidence) == (
        "accepted",
        "supported triage routing proposal",
        None,
    )


def test_proposal_from_another_agent_is_rejected():
    proposal = {
        "round_id": "round-1",
        "agent_id": "triager-2",
        "run_id": "run-1",
        "op": "route_issue",
        "evidence_event_ids": ["ev-1"],
    }
    run = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "role": "triager",
        "event_ids": ["ev-1"],
    }

    assert validate_proposal("round-1", proposal, run, {"ev-1": {"taint": 2}}) == (
        "rejected",
        "proposal run does not match its round and agent",
        None,
    )


def test_missing_or_unlinked_evidence_is_rejected():
    proposal = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "run_id": "run-1",
        "op": "route_issue",
        "evidence_event_ids": ["ev-missing"],
    }
    run = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "role": "triager",
        "event_ids": ["ev-1"],
    }

    assert validate_proposal("round-1", proposal, run, {}) == (
        "rejected",
        "proposal evidence is missing or not linked to its run",
        None,
    )


def test_action_with_only_untrusted_evidence_is_rejected():
    proposal = {
        "round_id": "round-1",
        "agent_id": "fixer-1",
        "run_id": "run-1",
        "op": "edit_file",
        "evidence_event_ids": ["ev-1"],
    }
    run = {
        "round_id": "round-1",
        "agent_id": "fixer-1",
        "role": "fixer",
        "event_ids": ["ev-1"],
    }

    assert validate_proposal("round-1", proposal, run, {"ev-1": {"taint": 2}}) == (
        "rejected",
        "action requires evidence with taint below 2",
        None,
    )


def test_evidence_with_invalid_taint_is_rejected():
    proposal = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "run_id": "run-1",
        "op": "route_issue",
        "evidence_event_ids": ["ev-1"],
    }
    run = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "role": "triager",
        "event_ids": ["ev-1"],
    }

    assert validate_proposal("round-1", proposal, run, {"ev-1": {"taint": 4}}) == (
        "rejected",
        "evidence event has invalid taint",
        None,
    )


def test_reproducer_bug_cause_requires_a_test_and_resolved_anchor():
    symbol = "scip-python python demo 1.0 `demo.module`/example()."
    proposal = {
        "_id": "prop-1",
        "round_id": "round-1",
        "agent_id": "reproducer-1",
        "run_id": "run-1",
        "op": "assert_fact",
        "evidence_event_ids": ["ev-1"],
        "payload": {
            "claim": "The function mishandles empty input",
            "kind": "bug_cause",
            "area": "demo",
            "anchor_names": [symbol],
            "confidence": 0.8,
            "decay_half_life_days": 30,
            "evidence_test": "tests/test_demo.py::test_empty_input",
            "supersedes": None,
        },
    }
    run = {
        "round_id": "round-1",
        "agent_id": "reproducer-1",
        "role": "reproducer",
        "event_ids": ["ev-1"],
    }
    evidence = {"ev-1": {"taint": 0}}
    index = ScipIndex(
        {symbol: (ScipDefinition(symbol, "demo.py", (0, 0, 1), None),)},
        sha="base-1",
    )

    status, _, fact = validate_proposal(
        "round-1",
        proposal,
        run,
        evidence,
        index=index,
        base_commit="base-1",
        committed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert status == "accepted"
    assert fact is not None
    assert fact["_id"] == "fact-00dfc6744588"
    assert fact["kind"] == "bug_cause"
    assert fact["anchors"] == [{"symbol": symbol, "sha": "base-1"}]
    assert fact["evidence_test"] == "tests/test_demo.py::test_empty_input"

    proposal["payload"]["evidence_test"] = None
    status, reason, fact = validate_proposal(
        "round-1",
        proposal,
        run,
        evidence,
        index=index,
        base_commit="base-1",
        committed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert (status, reason, fact) == (
        "rejected",
        "bug_cause facts require an evidence test",
        None,
    )


def test_unsupported_operation_is_rejected_even_with_trusted_evidence():
    proposal = {
        "round_id": "round-1",
        "agent_id": "fixer-1",
        "run_id": "run-1",
        "op": "edit_file",
        "evidence_event_ids": ["ev-1"],
    }
    run = {
        "round_id": "round-1",
        "agent_id": "fixer-1",
        "role": "fixer",
        "event_ids": ["ev-1"],
    }

    assert validate_proposal("round-1", proposal, run, {"ev-1": {"taint": 1}}) == (
        "rejected",
        "operation is not supported for this role",
        None,
    )
