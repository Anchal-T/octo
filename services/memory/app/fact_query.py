from __future__ import annotations

from typing import Any

from .dag import Dag, UnknownCommit


def valid_at(
    fact: dict[str, Any],
    closing_commits: set[str],
    dag: Dag,
    base_commit: str,
) -> bool:
    if base_commit not in dag.commits:
        raise UnknownCommit(base_commit)

    valid_from = fact.get("valid_from_commit")
    if not valid_from:
        return False
    if not dag.is_ancestor(valid_from, base_commit):
        return False

    all_closings = set(closing_commits)
    if fact.get("closing_commits"):
        all_closings.update(fact["closing_commits"])

    for closing_commit in all_closings:
        if dag.is_ancestor(closing_commit, base_commit):
            return False

    return True


def filter_valid(
    facts: list[dict[str, Any]],
    closures: dict[str, set[str]],
    dag: Dag,
    base_commit: str,
    *,
    ancestry_filter: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if base_commit not in dag.commits:
        raise UnknownCommit(base_commit)

    dropped = {"not_ancestor": 0, "closed": 0, "expired": 0}
    if not ancestry_filter:
        return list(facts), dropped

    valid: list[dict[str, Any]] = []
    for fact in facts:
        fact_id = fact.get("_id") or fact.get("id")
        closing = set(closures.get(fact_id, set()))
        if fact.get("closing_commits"):
            closing.update(fact["closing_commits"])

        valid_from = fact.get("valid_from_commit")
        if not valid_from or not dag.is_ancestor(valid_from, base_commit):
            dropped["not_ancestor"] += 1
            continue

        if any(dag.is_ancestor(c, base_commit) for c in closing):
            dropped["closed"] += 1
            continue

        valid.append(fact)

    return valid, dropped
