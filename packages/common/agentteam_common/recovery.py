import json
import re
from urllib.parse import quote

import httpx
from agentteam_contracts import IssueRunRequestV2, RoundRecord, RunStartAccepted

from .ids import issue_run_v2_prefix, stable_id, workflow_key

_INVOCATION_ID = re.compile(r"inv_[A-Za-z0-9_]{1,96}")


class RecoveryError(ValueError):
    pass


def _query(client: httpx.Client, admin_url: str, sql: str) -> list[dict]:
    response = client.post(
        f"{admin_url}/query",
        json={"query": sql},
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    rows = response.json()["rows"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RecoveryError("Restate returned an invalid recovery query result")
    return rows


def _invocation_id(value: str) -> str:
    if not isinstance(value, str) or not _INVOCATION_ID.fullmatch(value):
        raise RecoveryError("Expected a Restate invocation ID")
    return value


def _find_workflow(client: httpx.Client, admin_url: str, condition: str) -> dict | None:
    rows = _query(
        client,
        admin_url,
        "SELECT id, target_service_key, status, completion_result FROM sys_invocation "
        "WHERE target_service_name = 'IssueRunV2' AND target_handler_name = 'run' "
        f"AND {condition} LIMIT 2",
    )
    if len(rows) > 1:
        raise RecoveryError("Restate returned more than one IssueRunV2 invocation")
    return rows[0] if rows else None


def _find_origin(client: httpx.Client, admin_url: str, invocation_id: str) -> dict:
    invocation_id = _invocation_id(invocation_id)
    original = _find_workflow(client, admin_url, f"id = '{invocation_id}'")
    if original is None:
        raise RecoveryError("IssueRunV2 invocation was not found")
    return original


def _read_run_input(
    client: httpx.Client, admin_url: str, invocation_id: str
) -> IssueRunRequestV2 | None:
    invocation_id = _invocation_id(invocation_id)
    rows = _query(
        client,
        admin_url,
        f"SELECT entry_json FROM sys_journal WHERE id = '{invocation_id}' AND index = 0 LIMIT 1",
    )
    if not rows or rows[0].get("entry_json") is None:
        return None
    try:
        entry = json.loads(rows[0]["entry_json"])
        payload = entry["Command"]["Input"]["payload"]
        if not isinstance(payload, list):
            raise TypeError("expected a byte array")
        return IssueRunRequestV2.model_validate_json(bytes(payload))
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise RecoveryError(
            "Restate journal input could not be decoded as an IssueRunV2 request"
        ) from error


def _pin_recovery_inputs(
    client: httpx.Client, memory_url: str, previous: IssueRunRequestV2
) -> dict:
    if previous.base_commit is not None and previous.read_basis is not None:
        return {"base_commit": previous.base_commit, "read_basis": previous.read_basis}

    round_id = stable_id(
        "round", f"{issue_run_v2_prefix(previous.issue_id, previous.attempt)}:round:1"
    )
    response = client.get(f"{memory_url}/v1/rounds/{round_id}")
    if response.status_code == 404:
        raise RecoveryError("The failed run has no saved base and read basis")
    response.raise_for_status()
    original = RoundRecord.model_validate(response.json())
    if (
        original.id != round_id
        or original.seq != 1
        or original.base_commit is None
        or (
            previous.base_commit is not None
            and previous.base_commit != original.base_commit
        )
        or (
            previous.read_basis is not None
            and previous.read_basis != original.read_basis
        )
    ):
        raise RecoveryError(
            "The failed run's saved round conflicts with its original input"
        )
    return {"base_commit": original.base_commit, "read_basis": original.read_basis}


def _build_linked_attempt(
    client: httpx.Client,
    *,
    admin_url: str,
    memory_url: str,
    invocation_id: str,
    original: dict,
    attempt: int | None,
) -> IssueRunRequestV2:
    if original["status"] != "completed" or original["completion_result"] != "failure":
        raise RecoveryError(
            "Recovery requires a paused or terminal failed IssueRunV2 invocation"
        )
    previous = _read_run_input(client, admin_url, invocation_id)
    if previous is None:
        raise RecoveryError(
            "The failed invocation's original input is no longer retained"
        )
    if (
        workflow_key(previous.issue_id, previous.attempt)
        != original["target_service_key"]
    ):
        raise RecoveryError(
            "The failed invocation does not match its original issue and attempt"
        )
    if attempt is None or not previous.attempt < attempt <= 99:
        raise RecoveryError(
            "Supply a new --attempt number above the original attempt and at most 99"
        )

    github_writes = _query(
        client,
        admin_url,
        f"SELECT name FROM sys_journal WHERE id = '{invocation_id}' "
        "AND name LIKE 'open-pr:%' LIMIT 1",
    )
    if github_writes:
        raise RecoveryError(
            "The failed attempt reached PR dispatch. Inspect its GitHub receipts first."
        )

    if previous.recovery_from is not None:
        recovery_steps = _query(
            client,
            admin_url,
            f"SELECT name FROM sys_journal WHERE id = '{invocation_id}' "
            "AND name IN ('verify-recovery', 'record-recovery')",
        )
        # Recording starts only after verification returns successfully.
        if {row["name"] for row in recovery_steps} != {"verify-recovery", "record-recovery"}:
            raise RecoveryError("The failed recovery attempt was not verified and recorded")

    return IssueRunRequestV2.model_validate(
        {
            **previous.model_dump(),
            **_pin_recovery_inputs(client, memory_url, previous),
            "attempt": attempt,
            "recovery_from": {
                "invocation_id": invocation_id,
                "attempt": previous.attempt,
            },
        }
    )


def verify_linked_attempt(
    client: httpx.Client,
    *,
    admin_url: str,
    memory_url: str,
    request: IssueRunRequestV2,
) -> None:
    if request.recovery_from is None:
        raise RecoveryError("A linked attempt requires an original invocation")
    invocation_id = request.recovery_from.invocation_id
    expected = _build_linked_attempt(
        client,
        admin_url=admin_url,
        memory_url=memory_url,
        invocation_id=invocation_id,
        original=_find_origin(client, admin_url, invocation_id),
        attempt=request.attempt,
    )
    if request != expected:
        raise RecoveryError("Recovery input does not match the original invocation")


def recover_issue_run(
    client: httpx.Client,
    *,
    admin_url: str,
    memory_url: str,
    workflow_url: str,
    invocation_id: str,
    attempt: int | None,
    apply: bool,
) -> dict:
    original = _find_origin(client, admin_url, invocation_id)
    if original["status"] == "paused":
        if attempt is not None:
            raise RecoveryError(
                "Resume the paused invocation using its current attempt"
            )
        if apply:
            client.patch(
                f"{admin_url}/invocations/{invocation_id}/resume"
            ).raise_for_status()
        return {"action": "resume", "invocation_id": invocation_id, "applied": apply}

    retry = _build_linked_attempt(
        client,
        admin_url=admin_url,
        memory_url=memory_url,
        invocation_id=invocation_id,
        original=original,
        attempt=attempt,
    )
    key = workflow_key(retry.issue_id, retry.attempt)
    encoded_key = key.replace("'", "''")
    existing = _find_workflow(
        client, admin_url, f"target_service_key = '{encoded_key}'"
    )
    result = {
        "action": "new_attempt",
        "source_invocation_id": invocation_id,
        "workflow_key": key,
        "base_commit": retry.base_commit,
        "applied": False,
    }
    if existing is not None:
        if _read_run_input(client, admin_url, existing["id"]) != retry:
            raise RecoveryError(
                "The requested attempt is already in use by different or unverified input"
            )
        return {**result, "invocation_id": existing["id"], "already_submitted": True}
    if not apply:
        return result

    issue_id = quote(retry.issue_id, safe="")
    response = client.post(
        f"{workflow_url}/v2/issues/{issue_id}/start", json=retry.model_dump(mode="json")
    )
    response.raise_for_status()
    accepted = RunStartAccepted.model_validate(response.json())
    if (accepted.issue_id, accepted.attempt) != (retry.issue_id, retry.attempt):
        raise RecoveryError("The workflow service returned an unexpected attempt")
    submitted = _read_run_input(client, admin_url, accepted.invocation_id)
    if submitted is not None and submitted != retry:
        raise RecoveryError("The submitted attempt key was claimed by different input")
    return {
        **result,
        "invocation_id": accepted.invocation_id,
        "applied": True,
        "link_verified": submitted is not None,
    }
