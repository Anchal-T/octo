from eval.stale_facts.live import (
    DAILY_ESTIMATED_TOKEN_LIMIT,
    has_budget_for_next_request,
    load_checkpoint,
    next_live_run,
    planned_runs,
    save_checkpoint,
    should_pause_after_result,
)


def test_live_runs_are_interleaved_per_issue():
    runs = planned_runs()
    assert [(run["issue_number"], run["condition"]) for run in runs[:6]] == [
        (5, "C0"),
        (5, "C2"),
        (6, "C0"),
        (6, "C2"),
        (7, "C0"),
        (7, "C2"),
    ]
    assert len(runs) == 20


def test_checkpoint_resumes_first_unfinished_run():
    checkpoint = {
        "completed": {"5:C0": {"status": "failed"}},
        "active": None,
    }

    assert next_live_run(checkpoint)["run_key"] == "5:C2"
    checkpoint["active"] = {"run_key": "6:C0"}
    assert next_live_run(checkpoint)["run_key"] == "6:C0"


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


def test_daily_guard_stops_on_quota_or_provider_throttle():
    assert should_pause_after_result("quota_exhausted")
    assert should_pause_after_result("throttled")
    assert not should_pause_after_result("failed")


def test_daily_guard_reserves_the_next_call_without_crossing_limit():
    assert DAILY_ESTIMATED_TOKEN_LIMIT == 190_000
    assert has_budget_for_next_request(183_000, 7_000)
    assert not has_budget_for_next_request(183_001, 7_000)
