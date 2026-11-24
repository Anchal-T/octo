import asyncio

import pytest

from eval.stale_facts.live import (
    _USAGE_READ,
    DAILY_ESTIMATED_TOKEN_LIMIT,
    LEDGER_COUNTED_STATES,
    MODEL,
    PLAN_SIZE,
    REPO,
    _write_daily_status,
    counted_states_sql,
    has_budget_for_next_request,
    is_provider_failure,
    load_checkpoint,
    next_live_run,
    planned_runs,
    provider_failure_streak,
    run_day,
    save_checkpoint,
    should_pause_after_result,
    terminal_failure_result,
    verify_model_available,
)


def test_usage_gate_matches_ledger_quota_accounting():
    """HTTP-400 `refused` attempts consume no ledger quota; counting them in
    the slice's reserved total made the daily gate pause with budget free."""
    assert "refused" not in LEDGER_COUNTED_STATES
    assert set(LEDGER_COUNTED_STATES) == {
        "reserved",
        "completed",
        "throttled",
        "failed",
    }
    assert counted_states_sql() == "'reserved', 'completed', 'throttled', 'failed'"
    assert "__STATE_FILTER__" not in _USAGE_READ
    assert counted_states_sql() in _USAGE_READ


def test_live_runs_are_interleaved_per_issue():
    runs = planned_runs()
    assert [(run["issue_number"], run["condition"]) for run in runs] == [
        (5, "C0"),
        (5, "C2"),
        (6, "C0"),
        (6, "C2"),
        (7, "C0"),
        (7, "C2"),
        (8, "C0"),
        (8, "C2"),
        (9, "C0"),
        (9, "C2"),
    ]
    assert len(runs) == PLAN_SIZE == 10


def test_early_stop_counts_only_trailing_provider_failures():
    order = planned_runs()
    checkpoint = {
        "order": order,
        "completed": {
            "5:C0": {"status": "failed", "failure_reason": "repro_not_established"},
            "5:C2": {
                "status": "failed",
                "failure_reason": "repo agent failed: tool_call_validation_failed",
            },
            "6:C0": {"status": "failed", "failure_reason": "fix_not_established"},
        },
    }
    assert provider_failure_streak(checkpoint) == 0
    checkpoint["completed"]["6:C2"] = {
        "status": "failed",
        "failure_reason": "repo agent failed: tool_call_validation_failed",
    }
    assert is_provider_failure(checkpoint["completed"]["6:C2"])
    assert provider_failure_streak(checkpoint) == 1
    checkpoint["completed"]["7:C0"] = {
        "status": "failed",
        "failure_reason": "triager failed: provider_http_400",
    }
    assert provider_failure_streak(checkpoint) == 2
    # A later task miss resets the streak even if provider failures remain earlier.
    checkpoint["completed"]["7:C2"] = {
        "status": "failed",
        "failure_reason": "fix_not_established",
    }
    assert provider_failure_streak(checkpoint) == 0


def test_quota_and_throttle_pause_without_advancing_the_early_stop_streak():
    """A quota stop or an ordinary throttle pauses the invocation at once, but
    is a scheduling signal, not a provider-caused terminal failure."""
    quota = {"status": "quota_exhausted", "failure_reason": "day_tokens"}
    throttle = {"status": "throttled", "failure_reason": "minute_tokens"}
    assert should_pause_after_result("quota_exhausted")
    assert should_pause_after_result("throttled")
    assert not should_pause_after_result("awaiting_approval")
    assert not should_pause_after_result("failed")
    assert not is_provider_failure(quota)
    assert not is_provider_failure(throttle)
    checkpoint = {
        "order": planned_runs(),
        "completed": {"5:C0": quota, "5:C2": throttle},
    }
    assert provider_failure_streak(checkpoint) == 0
    # Even a whole streak of quota stops never reaches the two-failure stop.
    checkpoint["completed"]["6:C0"] = dict(quota)
    checkpoint["completed"]["6:C2"] = dict(throttle)
    assert provider_failure_streak(checkpoint) == 0


def test_generic_http_500_is_unknown_harness_not_a_provider_failure():
    """A bare workflow HTTP 500 is a harness error with no provider evidence;
    only a real provider failure marker supports provider attribution."""
    unknown = terminal_failure_result(500)
    assert unknown is not None
    assert unknown["status"] == "unknown"
    assert unknown["failure_source"] == "harness"
    assert unknown["success"] is False
    assert not is_provider_failure(unknown)
    # The old attribution (a bare 500 reason counted as provider evidence) is gone.
    assert not is_provider_failure(
        {"status": "failed", "failure_reason": "workflow returned HTTP 500"}
    )
    assert terminal_failure_result(503) is None
    # Unexplained HTTP 500 stops for inspection; it is not a provider failure.
    assert should_pause_after_result("unknown")
    assert not is_provider_failure(
        {"status": "unknown", "failure_reason": "workflow returned HTTP 500"}
    )


def test_real_provider_failure_markers_count_consecutively():
    for reason in (
        "tool_call_validation_failed",
        "repo agent failed: tool_call_validation_failed",
        "provider_http_400",
        "model_mismatch",
    ):
        assert is_provider_failure({"status": "failed", "failure_reason": reason})
    checkpoint = {
        "order": planned_runs(),
        "completed": {
            "5:C0": {
                "status": "failed",
                "failure_reason": "repo agent failed: tool_call_validation_failed",
            },
            "5:C2": {"status": "failed", "failure_reason": "provider_http_400"},
        },
    }
    assert provider_failure_streak(checkpoint) == 2


def test_checkpoint_resumes_first_unfinished_run():
    checkpoint = {
        "completed": {"5:C0": {"status": "failed"}},
        "active": None,
    }

    assert next_live_run(checkpoint)["run_key"] == "5:C2"
    checkpoint["active"] = {"run_key": "6:C0"}
    assert next_live_run(checkpoint)["run_key"] == "6:C0"
    checkpoint["active"] = None
    checkpoint["completed"].update(
        {"5:C2": {"status": "failed"}, "6:C0": {"status": "failed"}}
    )
    checkpoint["skipped"] = {"6:C2": "user-directed day 2 resume at #7/C0"}
    assert next_live_run(checkpoint)["run_key"] == "7:C0"


def test_checkpoint_atomically_preserves_active_and_completed_runs(tmp_path):
    checkpoint = {
        "schema_version": 1,
        "repo": "Anchal-T/sqlparse",
        "order": planned_runs(),
        "completed": {"5:C0": {"status": "failed"}},
        "active": {"run_key": "5:C2"},
    }
    path = tmp_path / "checkpoint.json"

    save_checkpoint(path, checkpoint)
    restored = load_checkpoint(path)

    assert next_live_run(restored) == {"run_key": "5:C2"}
    restored["completed"]["5:C2"] = {"status": "failed"}
    restored["active"] = None
    assert next_live_run(restored)["run_key"] == "6:C0"
    assert not path.with_suffix(".json.tmp").exists()


def test_model_availability_check_returns_only_pinned_listing(monkeypatch):
    from eval.stale_facts import live

    calls = []

    def fake_compose_exec(code, payload):
        calls.append((code, payload))
        return {"listed": True}

    monkeypatch.setattr(live, "_compose_exec", fake_compose_exec)

    assert verify_model_available()
    assert calls == [(live._MODEL_AVAILABILITY_CHECK, {"model": live.MODEL})]

    monkeypatch.setattr(live, "_compose_exec", lambda code, payload: {"listed": False})
    assert not verify_model_available()


def test_daily_guard_reserves_the_next_call_without_crossing_limit():
    assert DAILY_ESTIMATED_TOKEN_LIMIT == 190_000
    assert has_budget_for_next_request(183_000, 7_000)
    assert not has_budget_for_next_request(183_001, 7_000)


def test_skips_are_accounted_separately_from_completed_runs():
    from eval.stale_facts.live import plan_counts

    checkpoint = {
        "completed": {"5:C0": {}, "5:C2": {}},
        "skipped": {"6:C2": "user-directed"},
    }
    assert plan_counts(checkpoint) == (2, 1, 7)
    # A fully accounted plan with one skip must not claim all ten ran.
    full = {
        "completed": {
            run["run_key"]: {} for run in planned_runs() if run["run_key"] != "6:C2"
        },
        "skipped": {"6:C2": "user-directed"},
    }
    assert plan_counts(full) == (9, 1, 0)


def test_quota_pause_does_not_launch_later_slots(tmp_path, monkeypatch):
    """A quota_exhausted result stops the invocation immediately: no later
    slot is started and it does not trigger the two-provider-failure stop."""
    from eval.stale_facts import live

    day_key = "2026-11-16T00:00:00Z"

    def fake_usage(run_ids=None):
        return {
            "day_start": day_key,
            "requests": 1,
            "retries": 0,
            "model_failures": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "reserved_tokens": 1_000,
            "runs": {},
        }

    queried = []

    class FakeResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("unexpected raise_for_status in fake HTTP")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, path, params=None):
            queried.append(path)
            return FakeResponse(
                200, {"status": "quota_exhausted", "failure_reason": "day_tokens"}
            )

        async def post(self, path, json=None):
            raise AssertionError(
                "no workflow start should be needed for a direct status result"
            )

    monkeypatch.setattr(live, "read_model_usage", fake_usage)
    monkeypatch.setattr(
        live,
        "verify_live_config",
        lambda: {
            "max_call_reservation": 5_000,
            "groq_tpd": DAILY_ESTIMATED_TOKEN_LIMIT,
        },
    )
    monkeypatch.setattr(live.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(
        live,
        "_write_daily_status",
        lambda checkpoint, usage, reason: tmp_path / "status.md",
    )

    checkpoint = {
        "schema_version": 1,
        "repo": REPO,
        "model": MODEL,
        "order": planned_runs(),
        "issues": {
            str(number): {
                "issue_id": f"gh-1-{number}",
                "title": "t",
                "body": "b",
                "base_commit": "abcdef1234567890abcdef1234567890abcdef12",
                "fact_snapshot": [],
            }
            for number in (5, 6, 7, 8, 9)
        },
        "completed": {},
        "active": None,
        "days": {},
        "skipped": {"6:C2": "user-directed"},
        "model_checks": {day_key: {"listed": True, "checked_at": "x"}},
    }
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, checkpoint)

    result = asyncio.run(
        run_day(
            checkpoint_path=path,
            github_url="http://fake-gh",
            memory_url="http://fake-mem",
            workflow_url="http://fake-wf",
            poll_seconds=0,
        )
    )

    # Only the first slot's status was read; no later slot was started.
    assert queried == ["/v2/issues/gh-1-5/run"]
    assert "two consecutive" not in result["pause_reason"]
    assert "quota_exhausted" in result["pause_reason"]
    assert result["completed_total"] == 1
    assert result["skipped_total"] == 1
    assert result["pending_total"] == 8
    assert result["completed_this_invocation"] == 1
    restored = load_checkpoint(path)
    assert set(restored["completed"]) == {"5:C0"}
    assert restored["completed"]["5:C0"]["status"] == "quota_exhausted"
    assert restored["active"] is None


def test_daily_status_separates_quota_pauses_from_task_failures(tmp_path):
    day_key = "2026-11-16T00:00:00Z"

    def outcome(run_key, condition, **fields):
        return {
            "condition": condition,
            "day_key": day_key,
            "run_key": run_key,
            **fields,
        }

    checkpoint = {
        "days": {},
        "model_checks": {day_key: {"listed": True}},
        "skipped": {},
        "completed": {
            "5:C0": outcome("5:C0", "C0", success=True, status="awaiting_approval"),
            "5:C2": outcome(
                "5:C2",
                "C2",
                success=False,
                status="failed",
                failure_reason="repro_not_established",
            ),
            "6:C0": outcome(
                "6:C0",
                "C0",
                success=False,
                status="failed",
                failure_reason="repo agent failed: tool_call_validation_failed",
            ),
            "6:C2": outcome(
                "6:C2",
                "C2",
                success=False,
                status="quota_exhausted",
                failure_reason="day_tokens",
            ),
            "7:C0": outcome(
                "7:C0",
                "C0",
                success=False,
                status="throttled",
                failure_reason="minute_tokens",
            ),
            "7:C2": outcome(
                "7:C2",
                "C2",
                success=False,
                status="unknown",
                failure_reason="workflow returned HTTP 500",
                failure_source="harness",
            ),
        },
    }
    usage = {
        "day_start": day_key,
        "requests": 42,
        "retries": 3,
        "model_failures": 2,
        "tokens_in": 1_000,
        "tokens_out": 500,
        "reserved_tokens": 90_000,
    }
    path = _write_daily_status(
        checkpoint, usage, "quota_exhausted on 6:C2", output_dir=tmp_path
    )
    content = path.read_text()

    # A quota-only stop is a pause, not a task failure; a bare 500 stays unknown.
    assert "Failed runs today (non-quota terminal): **2**" in content
    assert "quota/throttle pauses: **2**" in content
    assert "unknown/harness stops (no provider attribution): **1**" in content
    assert "Task failures today" not in content
    assert (
        "Groq requests today (ledger-counted attempts, refused excluded): **42**"
        in content
    )


def test_final_all_accounted_summary_and_skips(tmp_path, monkeypatch):
    """When every runnable slot finishes, the final summary must say all slots
    are accounted for and keep the skip — not the starting pending count."""
    from eval.stale_facts import live

    day_key = "2026-11-16T00:00:00Z"

    def fake_usage(run_ids=None):
        return {
            "day_start": day_key,
            "requests": 1,
            "retries": 0,
            "model_failures": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "reserved_tokens": 1_000,
            "runs": {},
        }

    queried = []

    class FakeResponse:
        status_code = 202

        def json(self):
            return {
                "state": "awaiting_approval",
                "pr_number": 1,
                "pr_url": "http://pr/1",
                "head_sha": "abc1234",
            }

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, path, params=None):
            queried.append(path)
            return FakeResponse()

        async def post(self, path, json=None):
            raise AssertionError("no workflow start expected")

    monkeypatch.setattr(live, "read_model_usage", fake_usage)
    monkeypatch.setattr(
        live,
        "verify_live_config",
        lambda: {
            "max_call_reservation": 5_000,
            "groq_tpd": DAILY_ESTIMATED_TOKEN_LIMIT,
        },
    )
    monkeypatch.setattr(live.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(
        live,
        "_write_daily_status",
        lambda checkpoint, usage, reason: tmp_path / "status.md",
    )

    checkpoint = {
        "schema_version": 1,
        "repo": REPO,
        "model": MODEL,
        "order": planned_runs(),
        "issues": {
            str(number): {
                "issue_id": f"gh-1-{number}",
                "title": "t",
                "body": "b",
                "base_commit": "abcdef1234567890abcdef1234567890abcdef12",
                "fact_snapshot": [],
            }
            for number in (5, 6, 7, 8, 9)
        },
        "completed": {},
        "active": None,
        "days": {},
        "skipped": {"6:C2": "user-directed"},
        "model_checks": {day_key: {"listed": True, "checked_at": "x"}},
    }
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, checkpoint)

    result = asyncio.run(
        run_day(
            checkpoint_path=path,
            github_url="http://fake-gh",
            memory_url="http://fake-mem",
            workflow_url="http://fake-wf",
            poll_seconds=0,
        )
    )

    assert len(queried) == 9  # nine runnable slots, not the skipped one
    assert result["completed_total"] == 9
    assert result["skipped_total"] == 1
    assert result["pending_total"] == 0
    assert result["completed_this_invocation"] == 9
    assert "all 10 planned slots accounted for" in result["pause_reason"]
    assert "9 completed, 1 skipped" in result["pause_reason"]
    assert "pending" not in result["pause_reason"]
    restored = load_checkpoint(path)
    assert "6:C2" not in restored["completed"]
    assert restored["skipped"] == {"6:C2": "user-directed"}
    assert restored["active"] is None


def test_p1_preflight_rejects_strict_budget_mode(monkeypatch):
    from eval.stale_facts import live

    monkeypatch.setattr(live, "_compose_exec", lambda _: {"budget_mode": "strict"})
    with pytest.raises(RuntimeError, match="p1_estimate"):
        live.verify_live_config()
