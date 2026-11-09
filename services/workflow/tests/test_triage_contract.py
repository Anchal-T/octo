import pytest
from agentteam_contracts import IssueRunResult, TriageResult
from pydantic import ValidationError

IDS = {
    "issue_id": "issue-17",
    "round_id": "round-17",
    "event_id": "event-17",
    "run_id": "run-17",
    "proposal_id": "proposal-17",
}


def test_stored_result_without_triage_fields_still_validates():
    result = IssueRunResult.model_validate({**IDS, "status": "committed"})

    assert result.area == "unknown"
    assert result.triage is None
    assert result.triage_error is None


def test_committed_result_carries_triage():
    triage = TriageResult(
        kind="bug", priority="medium", summary="Example summary", area="fields"
    )
    result = IssueRunResult(**IDS, status="committed", area="fields", triage=triage)

    assert result.area == "fields"
    assert result.model_dump(mode="json")["triage"] == {
        "kind": "bug",
        "priority": "medium",
        "summary": "Example summary",
        "area": "fields",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "crash", "priority": "low", "summary": "Example"},
        {"kind": "bug", "priority": "urgent", "summary": "Example"},
        {"kind": "bug", "priority": "low", "summary": ""},
        {"kind": "bug", "priority": "low", "summary": "x" * 301},
        {"kind": "bug", "priority": "low", "summary": "Example"},
    ],
)
def test_triage_result_rejects_invalid_values(payload):
    with pytest.raises(ValidationError):
        TriageResult.model_validate(payload)


def test_triage_and_triage_error_are_exclusive():
    triage = TriageResult(kind="bug", priority="low", summary="Example", area="unknown")
    with pytest.raises(ValidationError):
        IssueRunResult(
            **IDS,
            status="committed",
            area="unknown",
            triage=triage,
            triage_error="unparseable",
        )


def test_only_committed_results_carry_triage_fields():
    with pytest.raises(ValidationError):
        IssueRunResult(
            **IDS,
            status="throttled",
            retry_in=5,
            area="unknown",
            triage_error="unparseable",
        )
