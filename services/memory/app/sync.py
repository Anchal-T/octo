import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from agentteam_common import is_safe_relative_path, is_test_path

from .scip import ScipIndex

Validation = tuple[Literal["accepted", "rejected"], str, dict[str, Any] | None]
ProposalDecision = tuple[
    dict[str, Any], Literal["accepted", "rejected"], str, dict[str, Any] | None
]
ACTION_OPS = frozenset({"edit_file", "grant_role"})
FACT_KINDS = {"triager": "user_report", "reproducer": "bug_cause"}


def resolve_anchors(
    anchor_names: Any,
    index: ScipIndex,
    base_commit: str | None,
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]] | None:
    if (
        not isinstance(anchor_names, list)
        or (not anchor_names and not allow_empty)
        or not base_commit
        or index.sha != base_commit
    ):
        return None

    anchors = []
    for name in anchor_names:
        if not isinstance(name, str):
            return None
        symbol = index.resolve(name)
        if symbol is None:
            return None
        anchors.append({"symbol": symbol, "sha": base_commit})
    return anchors


def validate_proposal(
    round_id: str,
    proposal: dict[str, Any],
    run: dict[str, Any] | None,
    evidence: Mapping[str, Mapping[str, Any] | None],
    *,
    index: ScipIndex | None = None,
    base_commit: str | None = None,
    committed_at: datetime | None = None,
    existing_fact_ids: set[str] | frozenset[str] | None = None,
) -> Validation:
    if (
        run is None
        or proposal.get("round_id") != round_id
        or run.get("round_id") != round_id
        or run.get("agent_id") != proposal.get("agent_id")
    ):
        return "rejected", "proposal run does not match its round and agent", None

    evidence_ids = list(proposal.get("evidence_event_ids") or [])
    run_event_ids = set(run.get("event_ids") or [])
    evidence_rows = [evidence.get(event_id) for event_id in evidence_ids]
    if (
        not evidence_ids
        or any(event_id not in run_event_ids for event_id in evidence_ids)
        or any(event is None for event in evidence_rows)
    ):
        return "rejected", "proposal evidence is missing or not linked to its run", None

    evidence_rows = [event for event in evidence_rows if event is not None]
    if any(
        type(event.get("taint")) is not int or not 0 <= event["taint"] <= 3
        for event in evidence_rows
    ):
        return "rejected", "evidence event has invalid taint", None

    if proposal.get("op") in ACTION_OPS and not any(
        type(event.get("taint")) is int and 0 <= event["taint"] < 2
        for event in evidence_rows
    ):
        return "rejected", "action requires evidence with taint below 2", None

    role = run.get("role")
    op = proposal.get("op")
    if op == "route_issue":
        if role == "triager":
            return "accepted", "supported triage routing proposal", None
        return "rejected", "operation is not supported for this role", None
    if op == "add_repro":
        if role != "reproducer":
            return "rejected", "operation is not supported for this role", None
        return _validate_add_repro(proposal.get("payload"), evidence_rows)
    if op == "edit_file":
        if role != "fixer":
            return "rejected", "operation is not supported for this role", None
        return _validate_edit_file(proposal.get("payload"))
    if op == "review":
        if role != "tester":
            return "rejected", "operation is not supported for this role", None
        return _validate_review(proposal.get("payload"), evidence_rows)
    if op != "assert_fact":
        return "rejected", "operation is not supported for this role", None

    fact_kind = FACT_KINDS.get(role) if isinstance(role, str) else None
    payload = proposal.get("payload")
    if not isinstance(payload, dict) or payload.get("kind") != fact_kind:
        return "rejected", "fact kind is not supported for this role", None
    claim = payload.get("claim")
    area = payload.get("area")
    if (
        not isinstance(claim, str)
        or not claim.strip()
        or not isinstance(area, str)
        or not area.strip()
    ):
        return "rejected", "fact requires a claim and area", None

    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        return "rejected", "fact confidence must be between 0 and 1", None

    half_life = payload.get("decay_half_life_days")
    if (
        isinstance(half_life, bool)
        or not isinstance(half_life, (int, float))
        or half_life <= 0
        or (isinstance(half_life, float) and not math.isfinite(half_life))
    ):
        return "rejected", "fact decay half-life must be positive", None

    evidence_test = payload.get("evidence_test")
    supersedes = payload.get("supersedes")
    if (evidence_test is not None and not isinstance(evidence_test, str)) or (
        supersedes is not None and not isinstance(supersedes, str)
    ):
        return "rejected", "fact optional fields must be strings", None
    if supersedes is not None and (
        existing_fact_ids is None or supersedes not in existing_fact_ids
    ):
        return "rejected", "supersedes target does not exist", None
    if role == "reproducer" and (
        not isinstance(evidence_test, str) or not evidence_test.strip()
    ):
        return "rejected", "bug_cause facts require an evidence test", None
    if (
        index is None
        or not base_commit
        or index.sha != base_commit
        or committed_at is None
        or committed_at.tzinfo is None
    ):
        return "rejected", "fact requires a matching index and base commit", None

    anchors = resolve_anchors(
        payload.get("anchor_names"),
        index,
        base_commit,
        allow_empty=fact_kind == "user_report",
    )
    if anchors is None:
        return "rejected", "fact anchors could not be resolved at the base commit", None

    fact_id = "fact-" + hashlib.sha1(proposal["_id"].encode("utf-8")).hexdigest()[:12]
    fact = {
        "_id": fact_id,
        "_valid_from": committed_at,
        "claim": claim,
        "kind": fact_kind,
        "area": area,
        "anchors": anchors,
        "provenance_event_ids": evidence_ids,
        "confidence": float(confidence),
        "decay_half_life_days": float(half_life),
        "last_confirmed_at": committed_at,
        "taint": max(event["taint"] for event in evidence_rows),
        "valid_from_commit": base_commit,
        "supersedes": supersedes,
        "evidence_test": evidence_test,
        "created_in_round": round_id,
        "proposal_id": proposal["_id"],
    }
    return "accepted", "supported triage user_report fact proposal", fact


def _valid_file_list(files: Any, *, must_be_test: bool) -> bool:
    if not isinstance(files, list) or not files:
        return False
    for path in files:
        if not isinstance(path, str) or not is_safe_relative_path(path):
            return False
        if is_test_path(path) != must_be_test:
            return False
    return True


def _validate_add_repro(
    payload: Any, evidence_rows: Sequence[Mapping[str, Any]]
) -> Validation:
    if not any(event.get("taint") == 0 for event in evidence_rows):
        return "rejected", "add_repro requires tool observation evidence", None
    if not isinstance(payload, dict):
        return "rejected", "add_repro requires a payload mapping", None
    patch = payload.get("patch")
    test_node_id = payload.get("test_node_id")
    test_passed = payload.get("test_passed")
    if not isinstance(patch, str) or not patch.strip():
        return "rejected", "add_repro requires a non-empty patch", None
    if not _valid_file_list(payload.get("files"), must_be_test=True):
        return "rejected", "add_repro files must be test paths only", None
    if not isinstance(test_node_id, str) or not test_node_id.strip():
        return "rejected", "add_repro requires a test_node_id", None
    if type(test_passed) is not bool or test_passed is not False:
        return "rejected", "Repro rejected by sync: no failing test.", None
    return "accepted", "supported reproducer add_repro proposal", None


def _validate_edit_file(payload: Any) -> Validation:
    if not isinstance(payload, dict):
        return "rejected", "edit_file requires a payload mapping", None
    patch = payload.get("patch")
    test_node_id = payload.get("test_node_id")
    test_passed = payload.get("test_passed")
    if not isinstance(patch, str) or not patch.strip():
        return "rejected", "edit_file requires a non-empty patch", None
    if not _valid_file_list(payload.get("files"), must_be_test=False):
        return "rejected", "edit_file files must be non-test source paths", None
    if not isinstance(test_node_id, str) or not test_node_id.strip():
        return "rejected", "edit_file requires a test_node_id", None
    if type(test_passed) is not bool or test_passed is not True:
        return "rejected", "edit_file requires passing repro test", None
    return "accepted", "supported fixer edit_file proposal", None


def _validate_review(
    payload: Any, evidence_rows: Sequence[Mapping[str, Any]]
) -> Validation:
    if not any(event.get("taint") == 0 for event in evidence_rows):
        return "rejected", "review requires clean-checkout tool evidence", None
    if not isinstance(payload, dict):
        return "rejected", "review requires a payload mapping", None
    verdict = payload.get("verdict")
    notes = payload.get("notes", "")
    repro_before = payload.get("repro_failed_before")
    repro_after = payload.get("repro_passed_after")
    suite_passed = payload.get("suite_passed")
    if verdict not in {"approve", "changes_requested"}:
        return "rejected", "review verdict must be approve or changes_requested", None
    if not isinstance(notes, str):
        return "rejected", "review notes must be a string", None
    if any(
        type(flag) is not bool for flag in (repro_before, repro_after, suite_passed)
    ):
        return "rejected", "review requires boolean clean-checkout test flags", None
    if verdict == "approve" and not (repro_before and repro_after and suite_passed):
        return (
            "rejected",
            "approve review requires passing clean-checkout checks",
            None,
        )
    return "accepted", "supported tester review proposal", None


def resolve_edit_conflicts(
    decisions: Sequence[ProposalDecision],
) -> list[ProposalDecision]:
    accepted_edits = [
        (idx, prop)
        for idx, (prop, status, _, _) in enumerate(decisions)
        if status == "accepted" and prop.get("op") == "edit_file"
    ]
    if len(accepted_edits) <= 1:
        return list(decisions)

    ordered = sorted(
        accepted_edits,
        key=lambda item: (
            not bool((item[1].get("payload") or {}).get("test_passed")),
            str(item[1].get("_id", "")),
        ),
    )
    claimed_files: set[str] = set()
    losers: set[int] = set()
    for idx, prop in ordered:
        files = set((prop.get("payload") or {}).get("files") or ())
        if files & claimed_files:
            losers.add(idx)
        else:
            claimed_files.update(files)

    return [
        (prop, "rejected", "conflict: requeued", None)
        if idx in losers
        else (prop, status, reason, fact)
        for idx, (prop, status, reason, fact) in enumerate(decisions)
    ]
