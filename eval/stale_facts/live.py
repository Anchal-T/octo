"""Resumable C0/C2 P1 live slice; all writes are scoped to Anchal-T/sqlparse."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from agentteam_common import stable_id
from agentteam_contracts import FactQuery, FactQueryResponse, IssueRunRequestV2

from services.workflow.app.fact_symbols import issue_references, symbol_candidates

REPO = "Anchal-T/sqlparse"
MODEL = "qwen/qwen3.8-27b"
DAILY_ESTIMATED_TOKEN_LIMIT = 190_000
ISSUE_BASES = {
    **{number: "60cdc649726bf1bc4f1b336050560b336da715ec" for number in (5, 6)},
    **{number: "09dbe1eec31ed1b3cd05366748865f11ba427d32" for number in range(7, 15)},
}
CONDITIONS = ("C0", "C2")
ATTEMPTS = {"C0": 90, "C2": 91}
DEFAULT_CHECKPOINT = Path("eval/.cache/stale-facts/p1-live-slice.json")
BODY_MAX_CHARS = 8_000

_PIN_CHECK = """
import json
from agentteam_common import Settings
from services.workflow.app.models.pools import pin_for
s = Settings.from_env()
roles = ("triager", "reproducer", "fixer", "tester")
pins = {role: pin_for(role, s) for role in roles}
print(json.dumps({
  "pins": {role: pins[role].model for role in roles},
  "groq_tpd": s.groq_tpd,
  "reset_tz": s.groq_reset_tz,
  "max_call_reservation": max(p.prompt_token_cap + p.max_tokens for p in pins.values()),
}))
"""

_MODEL_AVAILABILITY_CHECK = """
import json, sys
import httpx
from agentteam_common import Settings
from services.workflow.app.models.pools import pin_for, pool_for
requested = json.load(sys.stdin)
s = Settings.from_env()
pin = pin_for("triager", s)
pool = pool_for(pin, s)
listed = False
try:
  response = httpx.get(
    f"{pool.base_url}/models",
    headers={"Authorization": f"Bearer {s.groq_api_key}"},
    timeout=15,
  )
  if response.status_code == 200:
    payload = response.json()
    listed = any(
      isinstance(item, dict) and item.get("id") == requested["model"]
      for item in payload.get("data", [])
    )
except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError):
  pass
print(json.dumps({"listed": listed}))
"""

_USAGE_READ = """
import json, sqlite3, sys
from datetime import UTC, datetime
from zoneinfo import ZoneInfo
from agentteam_common import Settings
s = Settings.from_env()
requested = json.load(sys.stdin)
now = datetime.now(ZoneInfo(s.groq_reset_tz))
local_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
day_start = local_start.astimezone(UTC).isoformat().replace("+00:00", "Z")
db = sqlite3.connect(s.models_sqlite_path)
db.row_factory = sqlite3.Row
row = db.execute(
  "SELECT COUNT(*) AS requests, COALESCE(SUM(tokens_est), 0) AS reserved_tokens, "
  "COALESCE(SUM(CASE WHEN retry_ordinal > 1 THEN 1 ELSE 0 END), 0) AS retries "
  "FROM call_attempt WHERE pool = 'groq' AND day_window = ?", (day_start,)
).fetchone()
failures = db.execute(
  "SELECT COUNT(*) FROM call_log l JOIN call_attempt a "
  "ON a.request_id = l.request_id AND a.retry_ordinal = l.retry_ordinal "
  "WHERE a.pool = 'groq' AND a.day_window = ? "
  "AND l.outcome NOT IN ('completed', 'cached')", (day_start,)
).fetchone()[0]
tokens = db.execute(
  "SELECT COALESCE(SUM(c.tokens_in), 0) AS tokens_in, "
  "COALESCE(SUM(c.tokens_out), 0) AS tokens_out "
  "FROM completion c WHERE c.request_id IN "
  "(SELECT DISTINCT request_id FROM call_attempt WHERE pool = 'groq' AND day_window = ?)",
  (day_start,),
).fetchone()
runs = {}
for run_id in requested.get("run_ids", []):
  attempts = db.execute(
    "SELECT COUNT(*) AS requests, COALESCE(SUM(tokens_est), 0) AS reserved_tokens, "
    "COALESCE(SUM(CASE WHEN retry_ordinal > 1 THEN 1 ELSE 0 END), 0) AS retries "
    "FROM call_attempt WHERE pool = 'groq' AND run_id = ?", (run_id,)
  ).fetchone()
  run_failures = db.execute(
    "SELECT COUNT(*) FROM call_log WHERE run_id = ? "
    "AND outcome NOT IN ('completed', 'cached')", (run_id,)
  ).fetchone()[0]
  run_tokens = db.execute(
    "SELECT COALESCE(SUM(tokens_in), 0) AS tokens_in, "
    "COALESCE(SUM(tokens_out), 0) AS tokens_out FROM completion "
    "WHERE request_id IN (SELECT DISTINCT request_id FROM call_attempt "
    "WHERE pool = 'groq' AND run_id = ?)", (run_id,)
  ).fetchone()
  pinned = [r[0] for r in db.execute(
    "SELECT DISTINCT model_pinned FROM call_attempt WHERE run_id = ?", (run_id,)
  )]
  reported = [r[0] for r in db.execute(
    "SELECT DISTINCT model_reported FROM call_log "
    "WHERE run_id = ? AND model_reported IS NOT NULL", (run_id,)
  )]
  runs[run_id] = {
    "requests": attempts["requests"],
    "reserved_tokens": attempts["reserved_tokens"],
    "retries": attempts["retries"],
    "model_failures": run_failures,
    "tokens_in": run_tokens["tokens_in"],
    "tokens_out": run_tokens["tokens_out"],
    "models_pinned": pinned,
    "models_reported": reported,
  }
db.close()
print(json.dumps({
  "day_start": day_start,
  "requests": row["requests"],
  "reserved_tokens": row["reserved_tokens"],
  "retries": row["retries"],
  "model_failures": failures,
  "tokens_in": tokens["tokens_in"],
  "tokens_out": tokens["tokens_out"],
  "runs": runs,
}))
"""


def planned_runs() -> list[dict[str, Any]]:
    return [
        {
            "issue_number": number,
            "condition": condition,
            "run_key": f"{number}:{condition}",
        }
        for number in ISSUE_BASES
        for condition in CONDITIONS
    ]


def next_live_run(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    if checkpoint.get("active") is not None:
        return checkpoint["active"]
    completed = checkpoint.get("completed", {})
    return next(
        (run for run in planned_runs() if run["run_key"] not in completed), None
    )


def has_budget_for_next_request(
    used_tokens: int,
    next_request_tokens: int,
    limit: int = DAILY_ESTIMATED_TOKEN_LIMIT,
) -> bool:
    return used_tokens + next_request_tokens <= limit


def should_pause_after_result(status: str) -> bool:
    return status in {"quota_exhausted", "throttled"}


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text())
    if checkpoint.get("schema_version") != 1 or checkpoint.get("repo") != REPO:
        raise ValueError("checkpoint is not for this P1 live slice")
    if checkpoint.get("order") != planned_runs():
        raise ValueError("checkpoint run order does not match this slice")
    return checkpoint


def _compose_exec(code: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env",
            "--profile",
            "github",
            "-f",
            "deploy/compose.yaml",
            "-f",
            "deploy/compose.p1-live.yaml",
            "exec",
            "-T",
            "workflow",
            "python",
            "-c",
            code,
        ],
        cwd=root,
        input="" if payload is None else json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError("Could not read the workflow model ledger safely")
    try:
        return json.loads(process.stdout)
    except ValueError as error:
        raise RuntimeError("Workflow returned invalid model-ledger JSON") from error


def verify_live_config() -> dict[str, Any]:
    config = _compose_exec(_PIN_CHECK)
    if set(config["pins"].values()) != {MODEL}:
        raise RuntimeError("All four workflow roles must be pinned to the locked model")
    if config["groq_tpd"] != DAILY_ESTIMATED_TOKEN_LIMIT:
        raise RuntimeError(
            "Workflow GROQ_TPD must equal the 190,000-token P1 safety cap"
        )
    return config


def verify_model_available() -> bool:
    result = _compose_exec(_MODEL_AVAILABILITY_CHECK, {"model": MODEL})
    if not isinstance(result, dict) or not isinstance(result.get("listed"), bool):
        raise TypeError("Groq model availability check returned an invalid result")
    return result["listed"]


def read_model_usage(run_ids: list[str] | None = None) -> dict[str, Any]:
    return _compose_exec(_USAGE_READ, {"run_ids": run_ids or []})


async def _post_json(
    client: httpx.AsyncClient, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    response = await client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def _github_api_json(path: str) -> dict[str, Any]:
    process = subprocess.run(
        ["gh", "api", path, "--header", "Accept: application/vnd.github+json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError("Authenticated GitHub read failed for the fork")
    try:
        return json.loads(process.stdout)
    except ValueError as error:
        raise RuntimeError("GitHub CLI returned invalid JSON") from error


async def prepare_checkpoint(
    *,
    github_url: str,
    memory_url: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    async with (
        httpx.AsyncClient(base_url=github_url, timeout=30) as github,
        httpx.AsyncClient(base_url=memory_url, timeout=30) as memory,
    ):
        head_response = await github.get("/v1/repo/head")
        head_response.raise_for_status()
        if head_response.json().get("repo") != REPO:
            raise RuntimeError("GitHub service is not connected to Anchal-T/sqlparse")

        repo_metadata = _github_api_json(f"repos/{REPO}")
        if str(
            repo_metadata.get("full_name", "")
        ).casefold() != REPO.casefold() or not repo_metadata.get("fork"):
            raise RuntimeError("GitHub API target is not the Anchal-T/sqlparse fork")
        repo_id = int(repo_metadata["id"])

        frozen_issues: dict[str, Any] = {}
        for number, base_commit in ISSUE_BASES.items():
            issue_id = f"gh-{repo_id}-{number}"
            issue = _github_api_json(f"repos/{REPO}/issues/{number}")
            if issue.get("state") != "open":
                raise RuntimeError(f"Fork issue #{number} is not open")
            labels = {label.get("name") for label in issue.get("labels", [])}
            if "agent-team" not in labels:
                raise RuntimeError(
                    f"Fork issue #{number} is missing the agent-team label"
                )

            title = str(issue.get("title") or "")[:500]
            body = str(issue.get("body") or "")[:BODY_MAX_CHARS]
            text = title + "\n" + body
            candidates = symbol_candidates(base_commit, text)
            resolved = await _post_json(
                memory, "/v1/symbols/resolve", candidates.model_dump(mode="json")
            )
            query = FactQuery(
                symbols=resolved["symbols"],
                issue_refs=issue_references(REPO, text, issue_id=issue_id),
                base_commit=base_commit,
                ancestry_filter=True,
            )
            raw_facts = await _post_json(
                memory, "/v1/facts/query", query.model_dump(mode="json")
            )
            facts = FactQueryResponse.model_validate(raw_facts)
            if facts.base_commit != base_commit:
                raise RuntimeError(f"Fact snapshot base mismatch for issue #{number}")
            frozen_issues[str(number)] = {
                "issue_id": issue_id,
                "title": title,
                "body": body,
                "base_commit": base_commit,
                "fact_snapshot": [fact.model_dump(mode="json") for fact in facts.facts],
                "snapshot_dropped": facts.dropped,
            }

    return {
        "schema_version": 1,
        "repo": REPO,
        "model": MODEL,
        "runtime": runtime,
        "created_at": datetime.now(UTC).isoformat(),
        "order": planned_runs(),
        "issues": frozen_issues,
        "completed": {},
        "active": None,
        "days": {},
    }


def _model_run_ids(issue_id: str, attempt: int) -> list[str]:
    prefix = f"{issue_id}:v2:r{attempt}"
    keys = [f"{prefix}:triage:1", f"{prefix}:repro:1"]
    for fix_attempt in (1, 2):
        keys.extend((f"{prefix}:fix:{fix_attempt}", f"{prefix}:test:{fix_attempt}"))
    return [stable_id("run", key) for key in keys]


def _make_request(checkpoint: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    issue = checkpoint["issues"][str(run["issue_number"])]
    condition = run["condition"]
    request = IssueRunRequestV2(
        issue_id=issue["issue_id"],
        repo=REPO,
        title=issue["title"],
        body=issue["body"],
        attempt=ATTEMPTS[condition],
        base_commit=issue["base_commit"],
        fact_condition=condition,
        fact_snapshot=issue["fact_snapshot"] if condition == "C2" else None,
        pr_branch_suffix=f"p1-{condition.lower()}",
    )
    return request.model_dump(mode="json")


def _write_daily_status(
    checkpoint: dict[str, Any], usage: dict[str, Any], pause_reason: str
) -> Path:
    day_key = usage["day_start"]
    checkpoint["days"].setdefault(day_key, {"run_keys": []})
    today = [
        result
        for result in checkpoint["completed"].values()
        if result.get("day_key") == day_key
    ]
    completed_total = len(checkpoint["completed"])
    failures = sum(not result["success"] for result in today)
    retries = usage["retries"]
    model_check = checkpoint.get("model_checks", {}).get(day_key)
    model_check_count = int(model_check is not None)
    if model_check is None:
        model_listed = "not checked"
    else:
        model_listed = "yes" if model_check["listed"] else "no"
    content = "\n".join(
        [
            f"# P1 live slice — {day_key[:10]}",
            "",
            f"Repo: `{REPO}` only. Pinned model: `{MODEL}` for every role.",
            f"Runs completed today: **{len(today)}**; total: **{completed_total}/20**.",
            f"C0/C2 completed today: **{sum(r['condition'] == 'C0' for r in today)} / {sum(r['condition'] == 'C2' for r in today)}**.",
            f"Task failures today: **{failures}**.",
            f"Groq requests today: **{usage['requests']}**; retries: **{retries}**; model failures: **{usage['model_failures']}**.",
            f"Groq `/models` availability GETs today: **{model_check_count}**; `{MODEL}` listed: **{model_listed}** (separate from the model-call ledger).",
            f"Reported tokens today: **{usage['tokens_in']} input + {usage['tokens_out']} output**; estimated ledger reservations: **{usage['reserved_tokens']}**.",
            f"Daily reserved-token total: **{usage['reserved_tokens']} / {DAILY_ESTIMATED_TOKEN_LIMIT}**.",
            f"Pause reason: {pause_reason}.",
            "",
            "Checkpoint: `eval/.cache/stale-facts/p1-live-slice.json`. Draft PRs only; none approved or merged.",
            "",
        ]
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "runs"
        / f"P1-live-slice-{day_key[:10]}.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _check_run_models(stats: dict[str, Any]) -> None:
    for run in stats.values():
        if any(model != MODEL for model in run["models_pinned"]):
            raise RuntimeError("A run used a model other than the locked pin")
        if any(model != MODEL for model in run["models_reported"]):
            raise RuntimeError("Groq reported a model other than the locked pin")


async def run_day(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    github_url: str = "http://localhost:8003",
    memory_url: str = "http://localhost:8001",
    workflow_url: str = "http://localhost:8002",
    poll_seconds: float = 5.0,
    max_runs: int | None = None,
) -> dict[str, Any]:
    runtime = verify_live_config()
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("model") != MODEL:
            raise ValueError("checkpoint model does not match the locked pin")
    else:
        checkpoint = await prepare_checkpoint(
            github_url=github_url, memory_url=memory_url, runtime=runtime
        )
        save_checkpoint(checkpoint_path, checkpoint)

    if next_live_run(checkpoint) is not None:
        preflight_usage = read_model_usage()
        day_key = preflight_usage["day_start"]
        model_checks = checkpoint.setdefault("model_checks", {})
        if day_key not in model_checks:
            model_checks[day_key] = {
                "listed": verify_model_available(),
                "checked_at": datetime.now(UTC).isoformat(),
            }
            save_checkpoint(checkpoint_path, checkpoint)
        if not model_checks[day_key]["listed"]:
            pause_reason = (
                f"{MODEL} was not listed by Groq /models; no workflow run started"
            )
            usage = read_model_usage()
            status_path = _write_daily_status(checkpoint, usage, pause_reason)
            return {
                "completed_total": len(checkpoint["completed"]),
                "completed_this_invocation": 0,
                "pause_reason": pause_reason,
                "status_path": str(status_path),
                "usage": usage,
            }

    completed_this_invocation = 0
    pause_reason = "all 20 runs are complete"
    async with httpx.AsyncClient(base_url=workflow_url, timeout=30) as client:
        while run := next_live_run(checkpoint):
            usage = read_model_usage(
                _model_run_ids(
                    checkpoint["issues"][str(run["issue_number"])]["issue_id"],
                    ATTEMPTS[run["condition"]],
                )
            )
            day_key = usage["day_start"]
            day = checkpoint["days"].setdefault(day_key, {"run_keys": []})
            if run["run_key"] not in day["run_keys"]:
                day["run_keys"].append(run["run_key"])
            if checkpoint["active"] is None:
                if max_runs is not None and completed_this_invocation >= max_runs:
                    pause_reason = f"manual per-invocation limit ({max_runs})"
                    break
                if not has_budget_for_next_request(
                    usage["reserved_tokens"], runtime["max_call_reservation"]
                ):
                    pause_reason = "next call cannot be reserved below the 190,000-token daily safety limit"
                    break
                checkpoint["active"] = {
                    **run,
                    "request": _make_request(checkpoint, run),
                    "started_at": datetime.now(UTC).isoformat(),
                    "submitted_at": None,
                    "invocation_id": None,
                }
                save_checkpoint(checkpoint_path, checkpoint)

            active = checkpoint["active"]
            issue_id = active["request"]["issue_id"]
            attempt = active["request"]["attempt"]
            issue_path = quote(issue_id, safe="")
            while True:
                status = await client.get(
                    f"/v2/issues/{issue_path}/run", params={"attempt": attempt}
                )
                if status.status_code == 404:
                    if active["submitted_at"] is not None and active["invocation_id"]:
                        raise RuntimeError(
                            "Restate lost a checkpointed workflow invocation"
                        )
                    if active["submitted_at"] is not None:
                        await asyncio.sleep(poll_seconds)
                        continue
                    active["submitted_at"] = datetime.now(UTC).isoformat()
                    save_checkpoint(checkpoint_path, checkpoint)
                    try:
                        started = await client.post(
                            f"/v2/issues/{issue_path}/start", json=active["request"]
                        )
                    except httpx.HTTPError:
                        active["submitted_at"] = None
                        save_checkpoint(checkpoint_path, checkpoint)
                        raise
                    if started.status_code not in (200, 202, 409):
                        started.raise_for_status()
                    if started.status_code != 409:
                        active["invocation_id"] = started.json().get("invocation_id")
                    save_checkpoint(checkpoint_path, checkpoint)
                    await asyncio.sleep(poll_seconds)
                    continue
                if status.status_code == 202:
                    payload = status.json()
                    if payload.get("state") == "awaiting_approval":
                        result = {
                            "status": "awaiting_approval",
                            "success": True,
                            "pr_number": payload.get("pr_number"),
                            "pr_url": payload.get("pr_url"),
                            "head_sha": payload.get("head_sha"),
                        }
                        break
                    await asyncio.sleep(poll_seconds)
                    continue
                status.raise_for_status()
                payload = status.json()
                result = {
                    "status": payload.get("status", "failed"),
                    "success": False,
                    "failure_reason": payload.get("failure_reason"),
                    "pr_number": payload.get("pr_number"),
                    "pr_url": payload.get("pr_url"),
                    "head_sha": payload.get("pr_head_sha"),
                }
                break

            after = read_model_usage(_model_run_ids(issue_id, attempt))
            stats = after["runs"]
            _check_run_models(stats)
            completed = {
                **run,
                **result,
                "completed_at": datetime.now(UTC).isoformat(),
                "day_key": after["day_start"],
                "model_usage": stats,
            }
            checkpoint["completed"][run["run_key"]] = completed
            checkpoint["active"] = None
            save_checkpoint(checkpoint_path, checkpoint)
            completed_this_invocation += 1
            if should_pause_after_result(result["status"]):
                pause_reason = (
                    f"workflow returned {result['status']}; pausing further Groq calls"
                )
                break

        usage = read_model_usage()
    save_checkpoint(checkpoint_path, checkpoint)
    status_path = _write_daily_status(checkpoint, usage, pause_reason)
    return {
        "completed_total": len(checkpoint["completed"]),
        "completed_this_invocation": completed_this_invocation,
        "pause_reason": pause_reason,
        "status_path": str(status_path),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--github-url", default="http://localhost:8003")
    parser.add_argument("--memory-url", default="http://localhost:8001")
    parser.add_argument("--workflow-url", default="http://localhost:8002")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    result = asyncio.run(
        run_day(
            checkpoint_path=args.checkpoint,
            github_url=args.github_url,
            memory_url=args.memory_url,
            workflow_url=args.workflow_url,
            poll_seconds=args.poll_seconds,
            max_runs=args.max_runs,
        )
    )
    print(
        f"P1 live slice: {result['completed_total']}/20 completed; "
        f"{result['pause_reason']}. Daily usage: "
        f"{result['usage']['requests']} requests, "
        f"{result['usage']['reserved_tokens']} reserved tokens."
    )


if __name__ == "__main__":
    main()
