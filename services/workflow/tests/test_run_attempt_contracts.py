import pytest
from agentteam_common import workflow_key
from agentteam_contracts import (
    IssueRunRequest,
    IssueRunResult,
    RunStartAccepted,
    RunStatus,
)
from pydantic import ValidationError

IDS = {
    "issue_id": "gh-1-2",
    "round_id": "round-1",
    "event_id": "event-1",
    "run_id": "run-1",
    "proposal_id": "proposal-1",
}


def test_missing_attempt_defaults_to_one_on_all_run_contracts():
    request = IssueRunRequest(issue_id="gh-1-2", repo="bot/fork", title="Example")
    result = IssueRunResult.model_validate({**IDS, "status": "committed"})
    accepted = RunStartAccepted(issue_id="gh-1-2")
    status = RunStatus(issue_id="gh-1-2")

    assert request.attempt == 1
    assert result.attempt == 1
    assert accepted.attempt == 1
    assert status.attempt == 1


@pytest.mark.parametrize("attempt", [0, 100])
def test_attempt_rejects_values_outside_one_to_ninety_nine(attempt):
    with pytest.raises(ValidationError):
        IssueRunRequest(
            issue_id="gh-1-2", repo="bot/fork", title="Example", attempt=attempt
        )
    with pytest.raises(ValidationError):
        IssueRunResult.model_validate(
            {**IDS, "status": "committed", "attempt": attempt}
        )
    with pytest.raises(ValidationError):
        RunStartAccepted(issue_id="gh-1-2", attempt=attempt)
    with pytest.raises(ValidationError):
        RunStatus(issue_id="gh-1-2", attempt=attempt)


def test_workflow_key_keeps_attempt_one_and_suffixes_later_attempts():
    assert workflow_key("gh-1-2", 1) == "gh-1-2"
    assert workflow_key("gh-1-2", 2) == "gh-1-2-r2"


@pytest.mark.parametrize("attempt", [0, 100])
def test_workflow_key_rejects_attempt_outside_one_to_ninety_nine(attempt):
    with pytest.raises(ValueError):
        workflow_key("gh-1-2", attempt)
