from eval.stale_facts.live import (
    DAILY_ESTIMATED_TOKEN_LIMIT,
    LEDGER_COUNTED_STATES,
    PLAN_SIZE,
    _USAGE_READ,
    counted_states_sql,
    has_budget_for_next_request,
    is_provider_failure,
    provider_failure_streak,
    load_checkpoint,
    next_live_run,
    planned_runs,
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
                "failure_reason": "workflow returned HTTP 500",
            },
            "6:C0": {"status": "failed", "failure_reason": "fix_not_established"},
        },
    }
    assert provider_failure_streak(checkpoint) == 0
    checkpoint["completed"]["6:C2"] = {
        "status": "quota_exhausted",
        "failure_reason": "day_tokens",
    }
    assert is_provider_failure(checkpoint["completed"]["6:C2"])
    assert provider_failure_streak(checkpoint) == 1
    checkpoint["completed"]["7:C0"] = {
        "status": "throttled",
        "failure_reason": "minute_tokens",
    }
    assert provider_failure_streak(checkpoint) == 2
    # A later task miss resets the streak even if provider failures remain earlier.
    checkpoint["completed"]["7:C2"] = {
        "status": "failed",
        "failure_reason": "fix_not_established",
    }
    assert provider_failure_streak(checkpoint) == 0


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


def test_terminal_http_500_is_recorded_as_failed_without_retry():
    assert terminal_failure_result(500) == {
        "status": "failed",
        "success": False,
        "failure_reason": "workflow returned HTTP 500",
    }
    assert terminal_failure_result(503) is None
    assert should_pause_after_result("failed")


def test_daily_guard_stops_on_quota_or_provider_throttle():
    assert should_pause_after_result("quota_exhausted")
    assert should_pause_after_result("throttled")
    assert not should_pause_after_result("awaiting_approval")


def test_daily_guard_reserves_the_next_call_without_crossing_limit():
    assert DAILY_ESTIMATED_TOKEN_LIMIT == 190_000
    assert has_budget_for_next_request(183_000, 7_000)
    assert not has_budget_for_next_request(183_001, 7_000)
