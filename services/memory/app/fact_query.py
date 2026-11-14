from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .dag import Dag, UnknownCommit


@dataclass(frozen=True)
class FactExpiryConfig:
    user_report_ttl_commits: int = 50
    user_report_half_life_days: float = 30.0
    min_effective_confidence: float = 0.6
    user_report_initial_confidence: float = 0.8


def is_expired(
    fact: dict[str, Any],
    dag: Dag,
    base_commit: str,
    base_time: datetime,
    cfg: FactExpiryConfig,
) -> bool:
    if fact.get("kind") != "user_report":
        return False

    valid_from = fact.get("valid_from_commit")
    if not valid_from:
        return False

    # 1. Primary rule: hard TTL counted in first-parent commits on main
    path = dag.first_parent_path(base_commit)
    if valid_from in path:
        idx = path.index(valid_from)
        subpath = path[idx + 1 :]
        main_commits_count = sum(
            1 for sha in subpath if dag.commits[sha].branch == "main"
        )
        if main_commits_count >= cfg.user_report_ttl_commits:
            return True

    # 2. Secondary rule: confidence decay against base commit committer time
    last_confirmed_at = fact.get("last_confirmed_at")
    if last_confirmed_at is not None:
        if isinstance(last_confirmed_at, str):
            last_confirmed_at = datetime.fromisoformat(last_confirmed_at)
        t_base = base_time
        t_fact = last_confirmed_at
        if t_base.tzinfo is None and t_fact.tzinfo is not None:
            t_base = t_base.replace(tzinfo=UTC)
        elif t_base.tzinfo is not None and t_fact.tzinfo is None:
            t_fact = t_fact.replace(tzinfo=UTC)

        age_seconds = (t_base - t_fact).total_seconds()
        age_days = max(0.0, age_seconds / 86400.0)
        confidence = float(
            fact.get("confidence")
            if fact.get("confidence") is not None
            else cfg.user_report_initial_confidence
        )
        half_life = float(
            fact.get("decay_half_life_days")
            if fact.get("decay_half_life_days") is not None
            else cfg.user_report_half_life_days
        )
        eff_conf = confidence * (0.5 ** (age_days / half_life))
        if eff_conf < cfg.min_effective_confidence:
            return True

    return False


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
    cfg: FactExpiryConfig | None = None,
    base_time: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if base_commit not in dag.commits:
        raise UnknownCommit(base_commit)

    if cfg is None:
        cfg = FactExpiryConfig()

    if base_time is None:
        base_time = dag.commits[base_commit].committed_at

    dropped = {"not_ancestor": 0, "closed": 0, "expired": 0}
    valid: list[dict[str, Any]] = []

    for fact in facts:
        fact_id = fact.get("_id") or fact.get("id")
        closing = set(closures.get(fact_id, set()))
        if fact.get("closing_commits"):
            closing.update(fact["closing_commits"])

        if is_expired(fact, dag, base_commit, base_time, cfg):
            dropped["expired"] += 1
            continue

        if ancestry_filter:
            valid_from = fact.get("valid_from_commit")
            if not valid_from or not dag.is_ancestor(valid_from, base_commit):
                dropped["not_ancestor"] += 1
                continue

            if any(dag.is_ancestor(c, base_commit) for c in closing):
                dropped["closed"] += 1
                continue

        valid.append(fact)

    return valid, dropped
