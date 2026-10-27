from typing import Any, Literal

from .scip import ScipIndex

Validation = tuple[Literal["accepted", "rejected"], str]
ACTION_OPS = frozenset({"edit_file", "grant_role"})


def resolve_anchors(
    anchor_names: Any, index: ScipIndex, base_commit: str | None
) -> list[dict[str, str]] | None:
    if (
        not isinstance(anchor_names, list)
        or not anchor_names
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
) -> Validation:
    if (
        run is None
        or proposal.get("round_id") != round_id
        or run.get("round_id") != round_id
        or run.get("agent_id") != proposal.get("agent_id")
    ):
        return "rejected", "proposal run does not match its round and agent"

    evidence_ids = list(proposal.get("evidence_event_ids") or [])
    run_event_ids = set(run.get("event_ids") or [])
    evidence_rows = [evidence.get(event_id) for event_id in evidence_ids]
    if (
        not evidence_ids
        or any(event_id not in run_event_ids for event_id in evidence_ids)
        or any(event is None for event in evidence_rows)
    ):
        return "rejected", "proposal evidence is missing or not linked to its run"

    evidence_rows = [event for event in evidence_rows if event is not None]
    if any(
        type(event.get("taint")) is not int or not 0 <= event["taint"] <= 3
        for event in evidence_rows
    ):
        return "rejected", "evidence event has invalid taint"

    if proposal.get("op") in ACTION_OPS and not any(
        type(event.get("taint")) is int and 0 <= event["taint"] < 2
        for event in evidence_rows
    ):
        return "rejected", "action requires evidence with taint below 2"

    if run.get("role") != "triager" or proposal.get("op") != "route_issue":
        return "rejected", "operation is not supported for this role"

    return "accepted", "supported triage routing proposal"
