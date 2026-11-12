from datetime import UTC, datetime

from services.memory.app.scip import ScipDefinition, ScipIndex
from services.memory.app.sync import resolve_edit_conflicts, validate_proposal


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


def test_supersedes_must_reference_an_existing_fact():
    symbol = "scip-python python demo 1.0 `demo.module`/example()."
    proposal = {
        "_id": "prop-superseding",
        "round_id": "round-1",
        "agent_id": "triager-1",
        "run_id": "run-1",
        "op": "assert_fact",
        "evidence_event_ids": ["ev-1"],
        "payload": {
            "claim": "The report has been reconfirmed",
            "kind": "user_report",
            "area": "demo",
            "anchor_names": [symbol],
            "confidence": 0.8,
            "decay_half_life_days": 30,
            "evidence_test": None,
            "supersedes": "fact-previous",
        },
    }
    run = {
        "round_id": "round-1",
        "agent_id": "triager-1",
        "role": "triager",
        "event_ids": ["ev-1"],
    }
    evidence = {"ev-1": {"taint": 0}}
    index = ScipIndex(
        {symbol: (ScipDefinition(symbol, "demo.py", (0, 0, 1), None),)},
        sha="base-1",
    )
    options = {
        "index": index,
        "base_commit": "base-1",
        "committed_at": datetime(2026, 1, 1, tzinfo=UTC),
    }

    status, _, fact = validate_proposal(
        "round-1",
        proposal,
        run,
        evidence,
        existing_fact_ids={"fact-previous"},
        **options,
    )

    assert status == "accepted"
    assert fact is not None and fact["supersedes"] == "fact-previous"

    status, reason, fact = validate_proposal(
        "round-1",
        proposal,
        run,
        evidence,
        existing_fact_ids=set(),
        **options,
    )

    assert (status, reason, fact) == (
        "rejected",
        "supersedes target does not exist",
        None,
    )


def test_unsupported_operation_is_rejected_even_with_trusted_evidence():
    proposal = {
        "round_id": "round-1",
        "agent_id": "fixer-1",
        "run_id": "run-1",
        "op": "grant_role",
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


def test_reproducer_add_repro_requires_failing_test_and_test_only_files():
    run = {
        "round_id": "round-2",
        "agent_id": "reproducer",
        "role": "reproducer",
        "event_ids": ["ev-tool", "ev-llm"],
    }
    evidence = {"ev-tool": {"taint": 0}, "ev-llm": {"taint": 1}}
    valid = {
        "round_id": "round-2",
        "agent_id": "reproducer",
        "run_id": "run-repro",
        "op": "add_repro",
        "evidence_event_ids": ["ev-tool", "ev-llm"],
        "payload": {
            "patch": "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
            "files": ["tests/test_bug.py"],
            "test_node_id": "tests/test_bug.py::test_repro",
            "test_passed": False,
        },
    }
    assert validate_proposal("round-2", valid, run, evidence) == (
        "accepted",
        "supported reproducer add_repro proposal",
        None,
    )

    passing = {**valid, "payload": {**valid["payload"], "test_passed": True}}
    assert validate_proposal("round-2", passing, run, evidence)[0] == "rejected"

    non_test = {**valid, "payload": {**valid["payload"], "files": ["sqlparse/sql.py"]}}
    assert validate_proposal("round-2", non_test, run, evidence)[0] == "rejected"


def test_fixer_edit_file_requires_non_test_files_and_passing_repro():
    run = {
        "round_id": "round-3",
        "agent_id": "fixer",
        "role": "fixer",
        "event_ids": ["ev-tool", "ev-llm"],
    }
    evidence = {"ev-tool": {"taint": 0}, "ev-llm": {"taint": 1}}
    valid = {
        "round_id": "round-3",
        "agent_id": "fixer",
        "run_id": "run-fix",
        "op": "edit_file",
        "evidence_event_ids": ["ev-tool", "ev-llm"],
        "payload": {
            "patch": "diff --git a/sqlparse/lexer.py b/sqlparse/lexer.py\n",
            "files": ["sqlparse/lexer.py"],
            "test_node_id": "tests/test_bug.py::test_repro",
            "test_passed": True,
        },
    }
    assert validate_proposal("round-3", valid, run, evidence) == (
        "accepted",
        "supported fixer edit_file proposal",
        None,
    )

    touches_test = {
        **valid,
        "payload": {**valid["payload"], "files": ["tests/test_bug.py"]},
    }
    assert validate_proposal("round-3", touches_test, run, evidence)[0] == "rejected"


def test_tester_review_requires_taint_zero_and_passing_checks_on_approve():
    run = {
        "round_id": "round-4",
        "agent_id": "tester",
        "role": "tester",
        "event_ids": ["ev-tool", "ev-llm"],
    }
    evidence = {"ev-tool": {"taint": 0}, "ev-llm": {"taint": 1}}
    approved = {
        "round_id": "round-4",
        "agent_id": "tester",
        "run_id": "run-test",
        "op": "review",
        "evidence_event_ids": ["ev-tool", "ev-llm"],
        "payload": {
            "verdict": "approve",
            "notes": "",
            "repro_failed_before": True,
            "repro_passed_after": True,
            "suite_passed": True,
        },
    }
    assert validate_proposal("round-4", approved, run, evidence) == (
        "accepted",
        "supported tester review proposal",
        None,
    )

    broken_suite = {
        **approved,
        "payload": {**approved["payload"], "suite_passed": False},
    }
    assert validate_proposal("round-4", broken_suite, run, evidence)[0] == "rejected"

    changes_requested = {
        **approved,
        "payload": {
            **approved["payload"],
            "verdict": "changes_requested",
            "suite_passed": False,
            "notes": "Regression in test_grouping",
        },
    }
    assert (
        validate_proposal("round-4", changes_requested, run, evidence)[0] == "accepted"
    )

    no_tool_evidence = {**approved, "evidence_event_ids": ["ev-llm"]}
    assert (
        validate_proposal("round-4", no_tool_evidence, run, {"ev-llm": {"taint": 1}})[0]
        == "rejected"
    )


def test_resolve_edit_conflicts_keeps_deterministic_winner_per_file():
    p1 = {
        "_id": "prop-b",
        "op": "edit_file",
        "payload": {"files": ["sqlparse/lexer.py"], "test_passed": True},
    }
    p2 = {
        "_id": "prop-a",
        "op": "edit_file",
        "payload": {"files": ["sqlparse/lexer.py"], "test_passed": True},
    }
    decisions = [
        (p1, "accepted", "supported fixer edit_file proposal", None),
        (p2, "accepted", "supported fixer edit_file proposal", None),
    ]
    resolved = resolve_edit_conflicts(decisions)
    by_id = {prop["_id"]: (status, reason) for prop, status, reason, _ in resolved}
    assert by_id["prop-a"] == ("accepted", "supported fixer edit_file proposal")
    assert by_id["prop-b"] == ("rejected", "conflict: requeued")
