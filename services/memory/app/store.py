from __future__ import annotations

import hashlib
import json
from typing import Any

import psycopg
from agentteam_common import Settings, stable_id
from agentteam_contracts import EventWrite, ProposalWrite, RoundWrite, RunWrite
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg.types.string import StrDumperVarchar

from .sync import validate_proposal

TAINT_LABELS = {0: "trusted", 1: "internal", 2: "untrusted", 3: "quarantined"}
TABLES = {"events", "runs", "rounds", "proposals"}


class IdempotencyConflict(Exception):
    pass


class RoundConflict(Exception):
    pass


class MemoryStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(
            host=self.settings.xtdb_host,
            port=self.settings.xtdb_port,
            dbname=self.settings.xtdb_database,
            user=self.settings.xtdb_user,
            password=self.settings.xtdb_password or None,
            autocommit=True,
            row_factory=dict_row,
        )
        conn.adapters.register_dumper(str, StrDumperVarchar)
        return conn

    @staticmethod
    def _request_hash(request: Any) -> str:
        body = request.model_dump(mode="json", exclude={"idempotency_key"})
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _fetch(
        conn: psycopg.Connection[Any], table: str, record_id: str
    ) -> dict[str, Any] | None:
        if table not in TABLES:
            raise ValueError(f"unsupported table: {table}")
        query = sql.SQL("SELECT * FROM {} WHERE _id = %s").format(sql.Identifier(table))
        return conn.execute(query, (record_id,)).fetchone()

    def _write_once(
        self,
        table: str,
        record: dict[str, Any],
        request_hash: str,
        guard: str | None = None,
        guard_params: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        if table not in TABLES:
            raise ValueError(f"unsupported table: {table}")

        columns = list(record)
        insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )
        record_id = record["_id"]

        with self._connect() as conn:
            try:
                with conn.transaction():
                    conn.execute(
                        sql.SQL(
                            "ASSERT NOT EXISTS (SELECT 1 FROM {} WHERE _id = %s), "
                            "'record already exists'"
                        ).format(sql.Identifier(table)),
                        (record_id,),
                    )
                    if guard is not None:
                        conn.execute(guard, guard_params)
                    conn.execute(insert, tuple(record[column] for column in columns))
            except psycopg.Error as error:
                existing = None
                try:
                    existing = self._fetch(conn, table, record_id)
                except psycopg.Error:
                    pass
                if existing is None:
                    if guard is not None and "assert-failed" in str(error).lower():
                        raise RoundConflict("request requires an open round") from error
                    raise
                if existing.get("request_sha256") != request_hash:
                    raise IdempotencyConflict(
                        "idempotency key reused with a different request"
                    ) from error
                return existing

            row = self._fetch(conn, table, record_id)
            if row is None:
                raise RuntimeError(f"XTDB did not return the inserted {table} row")
            return row

    def ping(self) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 AS ready").fetchone()["ready"] == 1

    def create_event(self, request: EventWrite) -> dict[str, Any]:
        request_hash = self._request_hash(request)
        record_id = stable_id("ev", request.idempotency_key)
        record = {
            "_id": record_id,
            "request_sha256": request_hash,
            "run_id": None,
            "seq": 1,
            "parent_id": None,
            "source": request.source,
            "kind": request.kind,
            "taint": request.taint,
            "taint_label": TAINT_LABELS[request.taint],
            "subject_token": None,
            "pii_scrubbed": False,
            "payload": Jsonb(request.payload),
        }
        return self._write_once("events", record, request_hash)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._fetch(conn, "events", event_id)

    def get_round(self, round_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._fetch(conn, "rounds", round_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._fetch(conn, "runs", run_id)

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._fetch(conn, "proposals", proposal_id)

    def create_round(self, request: RoundWrite) -> dict[str, Any]:
        request_hash = self._request_hash(request)
        record_id = stable_id("round", request.idempotency_key)
        record = {
            "_id": record_id,
            "request_sha256": request_hash,
            "seq": request.seq,
            "status": "open",
            "base_commit": request.base_commit,
            "read_basis": request.read_basis,
            "committed_proposals": Jsonb([]),
        }
        return self._write_once("rounds", record, request_hash)

    def create_run(self, request: RunWrite) -> dict[str, Any]:
        request_hash = self._request_hash(request)
        record_id = stable_id("run", request.idempotency_key)
        record = {
            "_id": record_id,
            "request_sha256": request_hash,
            "agent_id": request.agent_id,
            "role": request.role,
            "round_id": request.round_id,
            "workflow_id": request.workflow_id,
            "step": request.step,
            "model": request.model,
            "base_commit": request.base_commit,
            "tokens_in": request.tokens_in,
            "tokens_out": request.tokens_out,
            "cost_usd": request.cost_usd,
            "outcome": request.outcome,
            "event_ids": Jsonb(request.event_ids),
            "started_at": request.started_at,
            "ended_at": request.ended_at,
        }
        guard = "ASSERT EXISTS (SELECT 1 FROM rounds WHERE _id = %s AND status = 'open'), 'run requires an open round'"
        return self._write_once(
            "runs", record, request_hash, guard, (request.round_id,)
        )

    def create_proposal(self, request: ProposalWrite) -> dict[str, Any]:
        request_hash = self._request_hash(request)
        record_id = stable_id("prop", request.idempotency_key)
        record = {
            "_id": record_id,
            "request_sha256": request_hash,
            "round_id": request.round_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
            "op": request.op,
            "status": "pending",
            "reason": None,
            "evidence_event_ids": Jsonb(request.evidence_event_ids),
            "payload": Jsonb(request.payload),
        }
        guard = "ASSERT EXISTS (SELECT 1 FROM rounds WHERE _id = %s AND status = 'open'), 'proposal requires an open round'"
        return self._write_once(
            "proposals", record, request_hash, guard, (request.round_id,)
        )

    def commit_round(self, round_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            try:
                with conn.transaction():
                    conn.execute(
                        "ASSERT EXISTS (SELECT 1 FROM rounds WHERE _id = %s AND status = 'open'), "
                        "'round is not open'",
                        (round_id,),
                    )
                    proposals = conn.execute(
                        "SELECT * FROM proposals WHERE round_id = %s AND status = 'pending'",
                        (round_id,),
                    ).fetchall()
                    if not proposals:
                        raise RoundConflict(
                            "a round needs at least one pending proposal"
                        )

                    accepted_ids = []
                    for proposal in proposals:
                        run = self._fetch(conn, "runs", proposal["run_id"])
                        evidence_ids = list(proposal.get("evidence_event_ids") or [])
                        evidence = {
                            event_id: self._fetch(conn, "events", event_id)
                            for event_id in evidence_ids
                        }
                        status, reason = validate_proposal(
                            round_id, proposal, run, evidence
                        )
                        conn.execute(
                            "UPDATE proposals SET status = %s, reason = %s WHERE _id = %s",
                            (status, reason, proposal["_id"]),
                        )
                        if status == "accepted":
                            accepted_ids.append(proposal["_id"])

                    conn.execute(
                        "UPDATE rounds SET status = 'committed', committed_proposals = %s WHERE _id = %s",
                        (Jsonb(accepted_ids), round_id),
                    )
            except psycopg.Error as error:
                existing = None
                try:
                    existing = self._fetch(conn, "rounds", round_id)
                except psycopg.Error:
                    pass
                if existing is not None and existing.get("status") == "committed":
                    return existing
                if "assert-failed" in str(error).lower():
                    raise RoundConflict("round is not open") from error
                raise

            row = self._fetch(conn, "rounds", round_id)
            if row is None:
                raise RuntimeError("XTDB did not return the committed round")
            return row
