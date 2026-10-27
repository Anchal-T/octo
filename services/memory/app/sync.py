import hashlib
import math
from datetime import datetime
from typing import Any, Literal

from .scip import ScipIndex

Validation = tuple[Literal["accepted", "rejected"], str, dict[str, Any] | None]
ACTION_OPS = frozenset({"edit_file", "grant_role"})


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
    evidence: dict[str, dict[str, Any] | None],
    *,
    index: ScipIndex | None = None,
    base_commit: str | None = None,
    committed_at: datetime | None = None,
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

    if run.get("role") != "triager":
        return "rejected", "operation is not supported for this role", None
    if proposal.get("op") == "route_issue":
        return "accepted", "supported triage routing proposal", None
    if proposal.get("op") != "assert_fact":
        return "rejected", "operation is not supported for this role", None

    payload = proposal.get("payload")
    if not isinstance(payload, dict) or payload.get("kind") != "user_report":
        return "rejected", "triager may only assert user_report facts", None
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
    if (
        index is None
        or not base_commit
        or index.sha != base_commit
        or committed_at is None
        or committed_at.tzinfo is None
    ):
        return "rejected", "fact requires a matching index and base commit", None

    anchors = resolve_anchors(
        payload.get("anchor_names"), index, base_commit, allow_empty=True
    )
    if anchors is None:
        return "rejected", "fact anchors could not be resolved at the base commit", None

    fact_id = "fact-" + hashlib.sha1(proposal["_id"].encode("utf-8")).hexdigest()[:12]
    fact = {
        "_id": fact_id,
        "_valid_from": committed_at,
        "claim": claim,
        "kind": "user_report",
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
