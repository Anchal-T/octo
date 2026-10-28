from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from agentteam_common import Settings, closure_id, stable_id
from agentteam_contracts import EventWrite, ProposalWrite, RoundWrite, RunWrite
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg.types.string import StrDumperVarchar

from .dag import Commit, UnknownCommit
from .scip_indexer import ScipIndexer
from .sync import validate_proposal

TAINT_LABELS = {0: "trusted", 1: "internal", 2: "untrusted", 3: "quarantined"}
TABLES = {"events", "runs", "rounds", "proposals", "commits"}


class IdempotencyConflict(Exception):
    pass


class RoundConflict(Exception):
    pass


class MemoryStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.scip_indexer: ScipIndexer | None = None
        if self.settings.scip_repo_path:
            if not self.settings.scip_project_name:
                raise ValueError("SCIP_PROJECT_NAME is required with SCIP_REPO_PATH")
            cache_dir = self.settings.scip_cache_dir or str(
                Path.home() / ".cache" / "agent-team" / "scip"
            )
            self.scip_indexer = ScipIndexer(
                self.settings.scip_repo_path,
                cache_dir,
                self.settings.scip_project_name,
            )

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

    def get_commit(self, sha: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._fetch(conn, "commits", sha)

    def get_commits(self) -> list[Commit]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT _id, parents, committed_at, branch, author, changed_symbols, "
                "merge_side_symbols FROM commits"
            ).fetchall()
        return [
            Commit(
                sha=row["_id"],
                parents=tuple(row["parents"] or ()),
                committed_at=row["committed_at"],
                branch=row["branch"],
                author=row["author"],
                changed_symbols=tuple(row["changed_symbols"] or ()),
                merge_side_symbols=tuple(
                    (item["parent"], tuple(item["symbols"]))
                    for item in row["merge_side_symbols"] or ()
                ),
            )
            for row in rows
        ]

    def facts_for_symbols(self, symbols: list[str]) -> list[dict[str, Any]]:
        if not symbols:
            return []
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in symbols)
        query = sql.SQL(
            "SELECT _id, valid_from_commit, anchors, evidence_test "
            "FROM facts FOR ALL VALID_TIME "
            "WHERE EXISTS (SELECT 1 FROM UNNEST(anchors) AS a(anc) "
            "WHERE (a.anc).symbol IN ({}))"
        ).format(placeholders)
        with self._connect() as conn:
            return list(conn.execute(query, symbols).fetchall())

    def facts_by_ids(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in fact_ids)
        query = sql.SQL(
            "SELECT _id, valid_from_commit, anchors, evidence_test "
            "FROM facts FOR ALL VALID_TIME WHERE _id IN ({})"
        ).format(placeholders)
        with self._connect() as conn:
            return list(conn.execute(query, fact_ids).fetchall())

    def closures_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in fact_ids)
        query = sql.SQL(
            "SELECT fact_id, closing_commit FROM fact_closures FOR ALL VALID_TIME "
            "WHERE fact_id IN ({})"
        ).format(placeholders)
        with self._connect() as conn:
            return list(conn.execute(query, fact_ids).fetchall())

    def all_fact_closures(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT fact_id, closing_commit FROM fact_closures FOR ALL VALID_TIME"
                ).fetchall()
            )

    def end_fact_valid_time(self, fact_ids: list[str], committed_at: datetime) -> None:
        if not fact_ids:
            return
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in fact_ids)
        query = sql.SQL(
            "DELETE FROM facts FOR PORTION OF VALID_TIME FROM %s TO NULL "
            "WHERE _id IN ({})"
        ).format(placeholders)
        with self._connect() as conn:
            conn.execute(query, (committed_at, *fact_ids))

    def record_fact_closure(
        self,
        fact_id: str,
        closing_commit: str,
        branch: str,
        closed_by_agent: str,
        reason: str,
        committed_at: datetime,
        evidence_test: str | None = None,
    ) -> str:
        record_id = closure_id(fact_id, closing_commit)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fact_closures (_id, _valid_from, fact_id, closing_commit, "
                "branch, closed_by_agent, reason, evidence_test) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    record_id,
                    committed_at,
                    fact_id,
                    closing_commit,
                    branch,
                    closed_by_agent,
                    reason,
                    evidence_test,
                ),
            )
        return record_id

    def record_commit(self, commit: Commit) -> dict[str, Any]:
        record = {
            "_id": commit.sha,
            "_valid_from": commit.committed_at,
            "parents": list(commit.parents),
            "committed_at": commit.committed_at,
            "branch": commit.branch,
            "author": commit.author,
            "changed_symbols": list(commit.changed_symbols),
            "merge_side_symbols": [
                {"parent": parent, "symbols": list(symbols)}
                for parent, symbols in commit.merge_side_symbols
            ],
        }
        with self._connect() as conn:
            for parent in commit.parents:
                if self._fetch(conn, "commits", parent) is None:
                    raise UnknownCommit(parent)
            try:
                with conn.transaction():
                    conn.execute(
                        "ASSERT NOT EXISTS (SELECT 1 FROM commits WHERE _id = %s), "
                        "'commit already exists'",
                        (commit.sha,),
                    )
                    for parent in commit.parents:
                        conn.execute(
                            "ASSERT EXISTS (SELECT 1 FROM commits WHERE _id = %s), "
                            "'commit parent is missing'",
                            (parent,),
                        )
                    conn.execute(
                        "INSERT INTO commits (_id, _valid_from, parents, committed_at, "
                        "branch, author, changed_symbols, merge_side_symbols) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            record["_id"],
                            record["_valid_from"],
                            Jsonb(record["parents"]),
                            record["committed_at"],
                            record["branch"],
                            record["author"],
                            Jsonb(record["changed_symbols"]),
                            Jsonb(record["merge_side_symbols"]),
                        ),
                    )
            except psycopg.Error as error:
                existing = self._fetch(conn, "commits", commit.sha)
                if existing is None:
                    for parent in commit.parents:
                        if self._fetch(conn, "commits", parent) is None:
                            raise UnknownCommit(parent) from error
                    raise
                fields = (
                    "_id",
                    "parents",
                    "committed_at",
                    "branch",
                    "author",
                    "changed_symbols",
                    "merge_side_symbols",
                )
                if all(existing.get(field) == record[field] for field in fields):
                    return existing
                raise IdempotencyConflict(
                    f"commit {commit.sha} was recorded with different data"
                ) from error

            row = self._fetch(conn, "commits", commit.sha)
            if row is None:
                raise RuntimeError("XTDB did not return the inserted commit")
            return row

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

    @staticmethod
    def _pending_proposals(
        conn: psycopg.Connection[Any], round_id: str
    ) -> list[dict[str, Any]]:
        return conn.execute(
            "SELECT * FROM proposals WHERE round_id = %s AND status = 'pending'",
            (round_id,),
        ).fetchall()

    def commit_round(self, round_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            state = self._fetch(conn, "rounds", round_id)
            if state is None:
                raise RoundConflict("round not found")
            if state.get("status") == "committed":
                return state
            if state.get("status") not in {"open", "syncing"}:
                raise RoundConflict("round cannot be synced from its current state")

            if state["status"] == "open":
                if not self._pending_proposals(conn, round_id):
                    raise RoundConflict("a round needs at least one pending proposal")
                try:
                    with conn.transaction():
                        conn.execute(
                            "ASSERT EXISTS (SELECT 1 FROM rounds WHERE _id = %s AND status = 'open'), "
                            "'round is not open'",
                            (round_id,),
                        )
                        conn.execute(
                            "UPDATE rounds SET status = 'syncing' WHERE _id = %s",
                            (round_id,),
                        )
                except psycopg.Error as error:
                    state = self._fetch(conn, "rounds", round_id)
                    if state is not None and state.get("status") == "committed":
                        return state
                    if state is None or state.get("status") != "syncing":
                        if "assert-failed" in str(error).lower():
                            raise RoundConflict("round is not open") from error
                        raise
                else:
                    state = self._fetch(conn, "rounds", round_id)
                    if state is not None and state.get("status") == "committed":
                        return state
                    if state is None or state.get("status") != "syncing":
                        raise RoundConflict("round did not enter syncing state")

            proposals = self._pending_proposals(conn, round_id)
            if not proposals:
                raise RoundConflict("a syncing round needs pending proposals")

            base_commit = state.get("base_commit")
            index = None
            committed_at = None
            if (
                self.scip_indexer is not None
                and base_commit
                and any(proposal.get("op") == "assert_fact" for proposal in proposals)
            ):
                try:
                    index = self.scip_indexer.load(base_commit)
                    committed_at = self.scip_indexer.committer_date(base_commit)
                except (OSError, ValueError, subprocess.CalledProcessError):
                    pass

            decisions = []
            accepted_ids = []
            for proposal in proposals:
                run = self._fetch(conn, "runs", proposal["run_id"])
                evidence_ids = list(proposal.get("evidence_event_ids") or [])
                evidence = {
                    event_id: self._fetch(conn, "events", event_id)
                    for event_id in evidence_ids
                }
                status, reason, fact = validate_proposal(
                    round_id,
                    proposal,
                    run,
                    evidence,
                    index=index,
                    base_commit=base_commit,
                    committed_at=committed_at,
                )
                decisions.append((proposal, status, reason, fact))
                if status == "accepted":
                    accepted_ids.append(proposal["_id"])

            try:
                with conn.transaction():
                    conn.execute(
                        "ASSERT EXISTS (SELECT 1 FROM rounds WHERE _id = %s AND status = 'syncing'), "
                        "'round is not syncing'",
                        (round_id,),
                    )
                    for proposal, status, _, fact in decisions:
                        conn.execute(
                            "ASSERT EXISTS (SELECT 1 FROM proposals "
                            "WHERE _id = %s AND round_id = %s AND status = 'pending'), "
                            "'proposal is no longer pending in this round'",
                            (proposal["_id"], round_id),
                        )
                        if status == "accepted" and fact is not None:
                            conn.execute(
                                "ASSERT NOT EXISTS (SELECT 1 FROM facts WHERE _id = %s), "
                                "'fact id already exists'",
                                (fact["_id"],),
                            )
                    for proposal, status, reason, _ in decisions:
                        conn.execute(
                            "UPDATE proposals SET status = %s, reason = %s WHERE _id = %s",
                            (status, reason, proposal["_id"]),
                        )
                    for _, status, _, fact in decisions:
                        if status == "accepted" and fact is not None:
                            conn.execute(
                                "INSERT INTO facts ("
                                "_id, _valid_from, claim, kind, area, anchors, "
                                "provenance_event_ids, confidence, decay_half_life_days, "
                                "last_confirmed_at, taint, valid_from_commit, supersedes, "
                                "evidence_test, created_in_round, proposal_id"
                                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                (
                                    fact["_id"],
                                    fact["_valid_from"],
                                    fact["claim"],
                                    fact["kind"],
                                    fact["area"],
                                    Jsonb(fact["anchors"]),
                                    Jsonb(fact["provenance_event_ids"]),
                                    fact["confidence"],
                                    fact["decay_half_life_days"],
                                    fact["last_confirmed_at"],
                                    fact["taint"],
                                    fact["valid_from_commit"],
                                    fact["supersedes"],
                                    fact["evidence_test"],
                                    fact["created_in_round"],
                                    fact["proposal_id"],
                                ),
                            )
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
                    raise RoundConflict(
                        "round is not syncing or a proposal changed"
                    ) from error
                raise

            row = self._fetch(conn, "rounds", round_id)
            if row is None:
                raise RuntimeError("XTDB did not return the committed round")
            return row
