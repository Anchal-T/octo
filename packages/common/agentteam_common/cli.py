from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from urllib.parse import quote

import httpx
from agentteam_contracts import ApprovalRequest, ApprovalResult

from .recovery import RecoveryError, recover_issue_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentteam")
    sub = parser.add_subparsers(dest="command", required=True)

    approve = sub.add_parser(
        "approve",
        help="Approve or reject an IssueRunV2 pull request (CLI fallback for D1).",
    )
    approve.add_argument("issue_id", help="Issue key, e.g. gh-1450220110-7")
    approve.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="Run attempt number (1..99, default 1)",
    )
    approve.add_argument(
        "--by",
        required=True,
        help="Maintainer login recording the decision",
    )
    approve.add_argument(
        "--reject",
        action="store_true",
        help="Reject the PR instead of approving",
    )
    approve.add_argument(
        "--reason",
        default=None,
        help="Optional reason (recorded on rejection)",
    )
    approve.add_argument(
        "--head-sha",
        default=None,
        help="Optional expected PR head commit SHA",
    )
    approve.add_argument(
        "--workflow-url",
        default=os.getenv("WORKFLOW_URL", "http://127.0.0.1:8002"),
        help="Workflow service base URL",
    )
    recover = sub.add_parser(
        "recover", help="Preview or recover a failed IssueRunV2 invocation."
    )
    recover.add_argument("invocation_id")
    recover.add_argument(
        "--attempt", type=int, help="New attempt number for a terminal failure"
    )
    recover.add_argument(
        "--apply", action="store_true", help="Apply the recovery preview"
    )
    recover.add_argument(
        "--admin-url", default=os.getenv("RESTATE_ADMIN_URL", "http://127.0.0.1:9070")
    )
    recover.add_argument(
        "--memory-url", default=os.getenv("MEMORY_URL", "http://127.0.0.1:8001")
    )
    recover.add_argument(
        "--workflow-url", default=os.getenv("WORKFLOW_URL", "http://127.0.0.1:8002")
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "recover":
        return _run_recovery(args, transport)

    request = ApprovalRequest(
        issue_id=args.issue_id,
        attempt=args.attempt,
        approved=not args.reject,
        by=args.by,
        reason=args.reason,
        head_sha=args.head_sha,
        source="cli",
    )
    base_url = str(args.workflow_url).rstrip("/")
    encoded_id = quote(request.issue_id, safe="")
    try:
        with httpx.Client(transport=transport, timeout=15.0) as client:
            response = client.post(
                f"{base_url}/v2/issues/{encoded_id}/approve",
                json=request.model_dump(mode="json"),
            )
    except httpx.HTTPError as error:
        print(f"Workflow request failed: {type(error).__name__}", file=sys.stderr)
        return 1

    if response.status_code >= 400:
        print(
            f"Workflow returned HTTP {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 1

    result = ApprovalResult.model_validate(response.json())
    if result.status == "stale_head":
        print(
            f"Stale head SHA for {result.issue_id} (attempt {result.attempt}); "
            "re-check the current PR head before approving.",
            file=sys.stderr,
        )
        return 1
    if result.status == "already_decided":
        print(
            f"Already decided for {result.issue_id} (attempt {result.attempt}, "
            f"stage={result.stage})."
        )
        return 0
    verb = "Approved" if request.approved else "Rejected"
    print(f"{verb} {result.issue_id} (attempt {result.attempt}, stage={result.stage}).")
    return 0


def _run_recovery(
    args: argparse.Namespace, transport: httpx.BaseTransport | None
) -> int:
    try:
        with httpx.Client(transport=transport, timeout=15, trust_env=False) as client:
            result = recover_issue_run(
                client,
                admin_url=args.admin_url.rstrip("/"),
                memory_url=args.memory_url.rstrip("/"),
                workflow_url=args.workflow_url.rstrip("/"),
                invocation_id=args.invocation_id,
                attempt=args.attempt,
                apply=args.apply,
            )
    except RecoveryError as error:
        print(f"Recovery stopped: {error}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as error:
        print(
            f"Recovery service returned HTTP {error.response.status_code}",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPError as error:
        print(f"Recovery service unavailable: {type(error).__name__}", file=sys.stderr)
        return 1
    except (TypeError, ValueError, KeyError, RecursionError):
        print("Recovery service returned invalid data", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
