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


def test_ten_real_git_commits_replay_closures_and_temporal_baseline(
    tmp_path, monkeypatch
):
    import os
    import subprocess

    from services.memory.app.git_repo import GitRepo
    from services.memory.app.repo import Repo
    from services.memory.app.time_linear_projector import TimeLinearProjector
    from services.memory.tests.test_repo import FakeIndexer, FakeStore, _index
    from services.workflow.app.models.client import ModelClient

    def no_model(*args, **kwargs):
        raise AssertionError("model call in retrieval-only git replay")

    monkeypatch.setattr(ModelClient, "call", no_model)
    path = tmp_path / "repo"
    path.mkdir()

    def git(*args, second=1):
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            "GIT_AUTHOR_DATE": f"2020-01-01T00:00:{second:02}+00:00",
            "GIT_COMMITTER_DATE": f"2020-01-01T00:00:{second:02}+00:00",
        }
        return subprocess.run(
            ["git", *args],
            cwd=path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet", "--initial-branch=main")
    main = []
    for n in range(1, 10):
        if n in (1, 3):
            (path / "module.py").write_text(
                f"def f():\n    return {1 if n == 1 else 2}\ndef g():\n    return 1\n"
            )
        (path / "README").write_text(f"Fixture commit {n}\n")
        git("add", ".")
        git(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            f"Fixture {n}",
            second=n,
        )
        main.append(git("rev-parse", "HEAD"))
    git("switch", "--quiet", "-c", "side", main[1])
    (path / "README").write_text("Unmerged side fixture\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Side", second=2)
    side = git("rev-parse", "HEAD")
    source = GitRepo(path)
    ordered = list(source.rev_list_parents())
    assert len(ordered) == 10
    facts = []
    for i, (origin, symbol) in enumerate(
        [(main[1], "example.f()."), (side, "example.f()."), (main[1], "example.g().")],
        1,
    ):
        facts.append(
            {
                "_id": f"fact-{i:012x}",
                "kind": "bug_cause",
                "claim": "Example seeded claim",
                "anchors": [{"symbol": symbol}],
                "valid_from_commit": origin,
                "confidence": 0.9,
                "decay_half_life_days": 90,
                "last_confirmed_at": source.committer_date(origin),
            }
        )
    store = FakeStore(facts)
    indexer = FakeIndexer(
        {sha: _index(sha, "example.f().", "example.g().") for sha in ordered}
    )
    Repo(store, source, indexer).replay(ordered, main_head=main[-1])
    dag = Dag.from_commits(store.get_commits())
    assert store.closures[0]["closing_commit"] == main[2]
    assert len(store.closures) == 1
    TimeLinearProjector(dag, store).end_for(facts[0]["_id"], main[2], main[-1])
    ends = {fid: when for ids, when in store.ended for fid in ids}
    snapshots = {
        base: [
            f
            for f in facts
            if dag.commits[f["valid_from_commit"]].committed_at
            <= dag.commits[base].committed_at
            and (
                f["_id"] not in ends or dag.commits[base].committed_at < ends[f["_id"]]
            )
        ]
        for base in [main[-1], side]
    }
    seeds = [
        {
            "id": str(i),
            "type": ty,
            "fact_ids": [facts[i - 1]["_id"]],
            "origin": facts[i - 1]["valid_from_commit"],
            "change": main[2] if i == 1 else None,
            "restore": None,
        }
        for i, ty in [(1, "behavior"), (2, "nonancestor"), (3, "control")]
    ]
    queries = [
        {"base": base, "symbols": ["example.f().", "example.g()."], "issue_refs": []}
        for base in snapshots
    ]
    result = evaluate(
        dag, facts, {facts[0]["_id"]: {main[2]}}, seeds, queries, snapshots
    )
    assert [result["conditions"][c]["injected"] for c in ("C1", "C2", "C3")] == [
        6,
        4,
        5,
    ]
    assert [result["conditions"][c]["stale_injected"] for c in ("C1", "C2", "C3")] == [
        2,
        0,
        1,
    ]
    assert result["invalidation"]["precision"] == result["invalidation"]["recall"] == 1
