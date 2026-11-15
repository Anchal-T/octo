import json
from datetime import UTC, datetime, timedelta

from eval.stale_facts.replay import evaluate
from services.memory.app.dag import Commit, Dag


def fixture():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    commits = [
        Commit(
            f"{i:040x}",
            () if i == 1 else (f"{i - 1:040x}",),
            start + timedelta(seconds=i),
            "main",
            None,
            (),
        )
        for i in range(1, 10)
    ]
    commits.append(
        Commit(
            "a" * 40,
            (f"{2:040x}",),
            start + timedelta(seconds=2, microseconds=1),
            "history",
            None,
            (),
        )
    )
    dag = Dag.from_commits(commits)
    facts = [
        {
            "_id": "fact-" + f"{i:012x}",
            "kind": "bug_cause",
            "claim": "Example seeded claim",
            "anchors": [{"symbol": "example.f()."}],
            "valid_from_commit": origin,
            "confidence": 0.9,
            "decay_half_life_days": 90,
            "last_confirmed_at": start,
        }
        for i, origin in [(1, f"{2:040x}"), (2, "a" * 40), (3, f"{2:040x}")]
    ]
    seeds = [
        {
            "id": str(i),
            "type": ty,
            "fact_ids": [facts[i - 1]["_id"]],
            "origin": facts[i - 1]["valid_from_commit"],
            "change": f"{3:040x}" if i == 1 else None,
            "restore": None,
        }
        for i, ty in [(1, "behavior"), (2, "nonancestor"), (3, "control")]
    ]
    queries = [
        {"base": base, "symbols": ["example.f()."], "issue_refs": []}
        for base in [f"{9:040x}", "a" * 40]
    ]
    snapshots = {f"{9:040x}": facts[1:], "a" * 40: facts}
    return dag, facts, seeds, queries, snapshots


def test_ten_commit_replay_matches_hand_computed_metrics_and_never_calls_models(
    monkeypatch,
):
    from services.workflow.app.models.client import ModelClient

    monkeypatch.setattr(
        ModelClient,
        "call",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("model call in retrieval replay")
        ),
    )
    dag, facts, seeds, queries, snapshots = fixture()
    result = evaluate(
        dag, facts, {facts[0]["_id"]: {f"{3:040x}"}}, seeds, queries, snapshots
    )
    assert result["conditions"]["C0"]["injected"] == 0
    assert result["conditions"]["C1"]["injected"] == 6
    assert result["conditions"]["C1"]["stale_injected"] == 2
    assert result["conditions"]["C2"]["injected"] == 4
    assert result["conditions"]["C2"]["stale_injected"] == 0
    assert result["conditions"]["C3"]["injected"] == 5
    assert result["conditions"]["C3"]["stale_injected"] == 1
    assert result["invalidation"]["precision"] == 1
    assert result["invalidation"]["recall"] == 1
    assert result["invalidation"]["delay_median"] == 0
    assert result["thresholds"]["T2"]["value"] == 1
    assert result["thresholds"]["T6"]["status"] == "FAIL"
    assert result["thresholds"]["T6"]["measured"] is False
    assert set(result["thresholds"]) == {f"T{i}" for i in range(1, 9)}
    assert json.dumps(result, sort_keys=True) == json.dumps(
        evaluate(
            dag, facts, {facts[0]["_id"]: {f"{3:040x}"}}, seeds, queries, snapshots
        ),
        sort_keys=True,
    )


def test_seed_branches_and_manifest_are_rebuildable_without_checkout_changes(tmp_path):
    from eval.stale_facts.seed import commit, construct, git, window

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "README").write_text("fixture\n")
    git(repo, "add", "README")
    git(repo, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "root")
    root = git(repo, "rev-parse", "HEAD")
    tip = root
    for i in range(60):
        tip = commit(repo, tip, {"README": f"fixture {i}\n"}, "main", i)
    win = window(repo, 60, 0)
    one = construct(repo, win, 1)
    two = construct(repo, win, 1)
    assert one == two
    assert git(repo, "rev-parse", "HEAD") == tip
    assert (repo / "README").read_text() == "fixture\n"
    assert len(one["seeds"]) == 125
    assert all(
        sum(s["type"] == ty for s in one["seeds"]) == 20
        for ty in ("rename", "move", "behavior", "revert", "nonancestor", "userreport")
    )
    assert all(b["name"].startswith("eval/p1-") for b in one["branches"])


def test_revert_reassertion_and_ttl_probes_do_not_change_shared_pool():
    from eval.stale_facts.oracle import Oracle

    dag, facts, seeds, queries, snapshots = fixture()
    seeds[0]["restore"] = f"{4:040x}"
    restored = {
        **facts[0],
        "_id": "fact-000000000004",
        "valid_from_commit": f"{4:040x}",
    }
    seeds[0]["fact_ids"].append(restored["_id"])
    facts.append(restored)
    queries = [
        {
            "base": f"{4:040x}",
            "seed_id": seeds[0]["id"],
            "symbols": ["example.f()."],
            "issue_refs": [],
        }
    ]
    snapshots = {f"{4:040x}": facts}
    result = evaluate(
        dag, facts, {facts[0]["_id"]: {f"{3:040x}"}}, seeds, queries, snapshots
    )
    assert result["revert_checks"] == {"passed": 1, "total": 1}
    assert result["thresholds"]["T5"]["status"] == "PASS"
    assert Oracle(dag, seeds).truth(facts[0], f"{4:040x}") == "valid"

    start = datetime(2020, 1, 1, tzinfo=UTC)
    ttl_dag = Dag(
        [
            Commit(
                f"{i:040x}",
                () if i == 1 else (f"{i - 1:040x}",),
                start + timedelta(days=i),
                "main",
                None,
                (),
            )
            for i in range(1, 52)
        ]
    )
    report = {
        "_id": "fact-aaaaaaaaaaaa",
        "kind": "user_report",
        "claim": "Synthetic still-true report",
        "anchors": [],
        "source_issue": "eval/sqlparse#1",
        "valid_from_commit": f"{1:040x}",
        "confidence": 0.8,
        "decay_half_life_days": 30,
        "last_confirmed_at": ttl_dag.commits[f"{1:040x}"].committed_at,
    }
    seed = {
        "id": "ttl",
        "type": "userreport",
        "fact_ids": [report["_id"]],
        "origin": f"{1:040x}",
        "change": None,
        "restore": None,
        "checks": [
            {"base": f"{n:040x}", "keep": n == 50, "isolate_ttl": True}
            for n in (50, 51)
        ],
    }
    before = dict(report)
    result = evaluate(ttl_dag, [report], {}, [seed], [], {})
    assert result["lifecycle_checks"]["passed"] == 2
    assert result["lifecycle_checks"]["natural_ttl49_already_decayed"] == 1
    assert result["thresholds"]["T8"]["status"] == "PASS"
    assert report == before
