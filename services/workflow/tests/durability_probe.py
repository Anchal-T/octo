"""Standalone, test-only Restate durability probe (V1a).

This module is *not* production code. It is a self-contained worker that proves,
through a real HTTP/Restate integration, that a workflow can survive a worker
restart, a durable timer, a durable promise, and a Restate container restart
without ever repeating a completed provider (model) call.

Run it with::

    python -m services.workflow.tests.durability_probe --port PORT --state-dir DIR

The ``--state-dir`` holds a stdlib SQLite database (``provider.sqlite``) that is
the fake-call counter. It records two kinds of facts separately:

* ``call_events``: one row *every time a fake model body actually executes*.
  Restate replay does not re-execute ``ctx.run_typed`` bodies, so a correct run
  leaves exactly one row per turn even if the worker/Restate restarts.
* ``completions``: the saved completion per stable request id. Kept separate so
  a body that short-circuits on a cache cannot hide a repeated execution.

No production imports, no env keys, no providers/models, no network calls to
anything but the Restate server that invokes this worker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import restate
from fastapi import FastAPI
from hypercorn.asyncio import serve
from hypercorn.config import Config

SERVICE_NAME = "DurabilityProbeV1"
DB_FILENAME = "provider.sqlite"
PROMISE_NAME = "approval"

_STATE_DIR: str | None = None

probe = restate.Workflow(SERVICE_NAME)


# --------------------------------------------------------------------------- #
# SQLite fake-call counter
# --------------------------------------------------------------------------- #
def _db_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / DB_FILENAME


def _connect(state_dir: str | Path) -> sqlite3.Connection:
    path = _db_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _init_db(state_dir: str | Path) -> None:
    conn = _connect(state_dir)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id  TEXT NOT NULL,
                turn        INTEGER NOT NULL,
                revision    TEXT NOT NULL,
                at_epoch    REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS completions (
                request_id     TEXT PRIMARY KEY,
                turn           INTEGER NOT NULL,
                revision       TEXT NOT NULL,
                output         TEXT NOT NULL,
                created_epoch  REAL NOT NULL
            )
            """
        )
    finally:
        conn.close()


def fake_model_turn(
    *,
    state_dir: str | Path,
    request_id: str,
    turn: int,
    revision: str,
) -> dict[str, Any]:
    """One fake provider turn. Runs only inside a journaled ``ctx.run_typed``.

    Always appends an execution event (so duplication is visible) and records a
    saved completion keyed by the stable request id (the "cache" whose hiding
    power the separate execution count defeats).
    """
    _init_db(state_dir)
    output = {
        "turn": turn,
        "request_id": request_id,
        "revision": revision,
        "text": f"fake-model-turn-{turn}",
        "marker": "durability-probe-v1",
    }
    payload = json.dumps(output, sort_keys=True)
    conn = _connect(state_dir)
    try:
        conn.execute(
            "INSERT INTO call_events(request_id, turn, revision, at_epoch) "
            "VALUES (?, ?, ?, ?)",
            (request_id, turn, revision, time.time()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO completions"
            "(request_id, turn, revision, output, created_epoch) "
            "VALUES (?, ?, ?, ?, ?)",
            (request_id, turn, revision, payload, time.time()),
        )
    finally:
        conn.close()
    return output


def evidence(state_dir: str | Path) -> dict[str, Any]:
    """Return sanitized counts and receipts from the SQLite fake-call counter.

    Interface (stable for the integration test)::

        {
          "calls": int,                # total body executions
          "keys": list[str],           # sorted request ids that executed
          "call_counts": dict,         # request id -> execution count
          "completions": int,          # saved completions
          "completion_keys": list[str],
          "events": [ {request_id, turn, revision, at_epoch}, ... ],
          "receipts": [ {request_id, turn, revision, output, created_epoch}, ... ],
        }
    """
    _init_db(state_dir)
    conn = _connect(state_dir)
    try:
        event_rows = conn.execute(
            "SELECT request_id, turn, revision, at_epoch FROM call_events ORDER BY id"
        ).fetchall()
        completion_rows = conn.execute(
            "SELECT request_id, turn, revision, output, created_epoch "
            "FROM completions ORDER BY request_id"
        ).fetchall()
    finally:
        conn.close()

    call_counts: dict[str, int] = {}
    for request_id, _turn, _revision, _at in event_rows:
        call_counts[request_id] = call_counts.get(request_id, 0) + 1

    return {
        "calls": len(event_rows),
        "keys": sorted(call_counts),
        "call_counts": call_counts,
        "completions": len(completion_rows),
        "completion_keys": sorted(row[0] for row in completion_rows),
        "events": [
            {
                "request_id": row[0],
                "turn": row[1],
                "revision": row[2],
                "at_epoch": row[3],
            }
            for row in event_rows
        ],
        "receipts": [
            {
                "request_id": row[0],
                "turn": row[1],
                "revision": row[2],
                "output": json.loads(row[3]),
                "created_epoch": row[4],
            }
            for row in completion_rows
        ],
    }


# --------------------------------------------------------------------------- #
# Workflow handlers
# --------------------------------------------------------------------------- #
def _configured_state_dir() -> str:
    if _STATE_DIR is None:
        raise RuntimeError("durability probe state dir is not configured")
    return _STATE_DIR


@probe.main(name="run")
async def run(ctx: restate.WorkflowContext, request: dict) -> dict:
    """Turn 1 -> durable sleep -> turn 2 -> durable approval -> done."""
    key = ctx.key()
    revision = str(request["revision"])
    timer_seconds = float(request["timer_seconds"])

    ctx.set("revision", revision)
    ctx.set("stage", "running")

    turn_one = await ctx.run_typed(
        f"model:{key}:turn:1",
        fake_model_turn,
        restate.RunOptions(type_hint=dict),
        state_dir=_configured_state_dir(),
        request_id=f"{key}:{revision}:turn:1",
        turn=1,
        revision=revision,
    )
    ctx.set("stage", "first_turn_done")
    ctx.set("stage", "timer_wait")
    await ctx.sleep(timedelta(seconds=timer_seconds), name=f"durable-sleep:{key}")

    turn_two = await ctx.run_typed(
        f"model:{key}:turn:2",
        fake_model_turn,
        restate.RunOptions(type_hint=dict),
        state_dir=_configured_state_dir(),
        request_id=f"{key}:{revision}:turn:2",
        turn=2,
        revision=revision,
    )
    ctx.set("stage", "second_turn_done")
    ctx.set("stage", "waiting_approval")

    decision = await ctx.promise(PROMISE_NAME, type_hint=dict).value()
    ctx.set("stage", "done")
    return {
        "revision": revision,
        "turns": [turn_one, turn_two],
        "decision": decision,
        "approved": bool(decision["approved"]),
    }


@probe.handler(name="info")
async def info(ctx: restate.WorkflowSharedContext) -> dict:
    """Readiness/stage view for the integration test."""
    return {
        "stage": await ctx.get("stage", type_hint=str) or "running",
        "revision": await ctx.get("revision", type_hint=str),
    }


@probe.handler(name="approve")
async def approve(ctx: restate.WorkflowSharedContext, request: dict) -> dict:
    """Resolve the durable promise; stale and duplicate approvals are safe."""
    stored_revision = await ctx.get("revision", type_hint=str)
    if request.get("revision") != stored_revision:
        return {"status": "stale_revision"}

    promise = ctx.promise(PROMISE_NAME, type_hint=dict)
    if await promise.peek() is not None:
        return {"status": "already_decided"}

    decision = {
        "revision": request["revision"],
        "approved": bool(request["approved"]),
        "by": str(request["by"]),
    }
    try:
        await promise.resolve(decision)
    except restate.TerminalError as error:
        if error.status_code == 409 or "already completed" in str(error).lower():
            return {"status": "already_decided"}
        raise
    return {"status": "recorded"}


# --------------------------------------------------------------------------- #
# HTTP worker
# --------------------------------------------------------------------------- #
def build_app(state_dir: str | Path) -> FastAPI:
    """Build the FastAPI worker: health, evidence, and the Restate endpoint."""
    global _STATE_DIR
    _STATE_DIR = str(state_dir)
    _init_db(_STATE_DIR)

    app = FastAPI(title="Durability Probe V1 (test only)")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": SERVICE_NAME}

    @app.get("/evidence")
    def get_evidence() -> dict[str, Any]:
        return evidence(_STATE_DIR)

    app.mount(
        "/restate/v1",
        restate.app(services=[probe], protocol="request_response"),
    )
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)

    config = Config()
    config.bind = [f"0.0.0.0:{args.port}"]
    config.accesslog = None
    asyncio.run(serve(build_app(args.state_dir), config))


if __name__ == "__main__":
    main()
