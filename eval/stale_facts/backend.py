"""Run as a Memory-owned maintenance process against an isolated XTDB instance.

All fact writes use MemoryStore's normal round/proposal transaction. No eval SQL
writer is introduced; SQL below is read-only snapshot access for the baseline.
"""

from datetime import datetime
from hashlib import sha1

from agentteam_contracts import EventWrite, ProposalWrite, RoundWrite, RunWrite

from services.memory.app.dag import Commit, Dag
from services.memory.app.git_repo import GitRepo
from services.memory.app.repo import Repo
from services.memory.app.time_linear_projector import TimeLinearProjector

from .seed import check_seed, git


def metadata(store, repo, head):
    main = set(git(repo, "rev-list", "--first-parent", head).splitlines())
    graph = GitRepo(repo).rev_list_parents()
    info = {}
    for line in git(repo, "log", "--all", "--format=%H%x00%cI%x00%an").splitlines():
        sha, date, author = line.split("\x00", 2)
        info[sha] = (datetime.fromisoformat(date), author)
    commits = [
        Commit(
            sha,
            parents,
            info[sha][0],
            "main" if sha in main else "history",
            info[sha][1],
            (),
            symbols_indexed=False,
        )
        for sha, parents in graph.items()
    ]
    store.record_commit_metadata(commits)
    return Dag.from_commits(commits)


def populate(store, manifest):
    for reassert in (False, True):
        for row in manifest["seeds"]:
            if reassert and not row["restore"]:
                continue
            base = row["restore"] if reassert else row["origin"]
            key = f"p1-eval:{manifest['seed']}:{row['id']}:{'reassert' if reassert else 'initial'}"
            role = "triager" if row["type"] == "userreport" else "reproducer"
            checked = store.scip_indexer.committer_date(base)
            rnd = store.create_round(
                RoundWrite(
                    idempotency_key=key, seq=1, base_commit=base, read_basis=checked
                )
            )
            event = store.create_event(
                EventWrite(
                    idempotency_key=key + ":evidence",
                    source="tool" if role == "reproducer" else "github_issue",
                    kind="observation",
                    taint=0 if role == "reproducer" else 2,
                    payload={
                        "purpose": "explicitly synthetic / historical P1 seed, not a model run",
                        "tool": "python_assert",
                        "seed_id": row["id"],
                        "predicate": row.get("evidence"),
                        "checked_commit": base,
                        "verified": True,
                    },
                )
            )
            run = store.create_run(
                RunWrite(
                    idempotency_key=key + ":run",
                    agent_id=role,
                    role=role,
                    round_id=rnd["_id"],
                    workflow_id=key,
                    step="eval_seed",
                    base_commit=base,
                    outcome="seeded",
                    event_ids=[event["_id"]],
                    started_at=checked,
                    ended_at=checked,
                )
            )
            payload = {
                "kind": "user_report" if role == "triager" else "bug_cause",
                "claim": row["claim"],
                "area": "parser-tree",
                "anchor_names": row["anchor_names"],
                "confidence": 0.8 if role == "triager" else 0.9,
                "decay_half_life_days": 30 if role == "triager" else 90,
                "evidence_test": None
                if role == "triager"
                else f"eval/stale_facts/seed.py::check_seed[{row['id']}]",
                "source_issue": row.get("source_issue"),
                "supersedes": row["fact_ids"][0] if reassert else None,
            }
            prop = store.create_proposal(
                ProposalWrite(
                    idempotency_key=key + ":assert",
                    round_id=rnd["_id"],
                    agent_id=role,
                    run_id=run["_id"],
                    op="assert_fact",
                    evidence_event_ids=[event["_id"]],
                    payload=payload,
                )
            )
            result = store.commit_round(rnd["_id"])
            if prop["_id"] not in result["committed_proposals"]:
                raise RuntimeError(store.get_proposal(prop["_id"]))
            fid = "fact-" + sha1(prop["_id"].encode()).hexdigest()[:12]
            if not reassert:
                row["fact_ids"] = []
            row["fact_ids"].append(fid)
            index = store.scip_indexer.load(base)
            row["symbols"] = [index.resolve(name) for name in row["anchor_names"]]
            assert all(row["symbols"])


def execute(store, repo, manifest, progress=print):
    dag = metadata(store, repo, manifest["window"]["head"])
    progress("Creating validated fact proposals", flush=True)
    populate(store, manifest)
    selected = set(
        manifest["window"]["main"]
        + manifest["window"]["side"]
        + manifest["synthetic_commits"]
    )
    ordered = dag.topological_order(selected)
    processor = Repo(store, GitRepo(repo), store.scip_indexer)
    # A single Repo instance keeps decoded indexes between chunks. Each chunk
    # still uses the real parents-first replay, including merge-second-parent diffs.
    for offset in range(0, len(ordered), 25):
        processor.replay(
            ordered[offset : offset + 25], main_head=manifest["window"]["head"]
        )
        progress(
            f"Replayed {min(offset + 25, len(ordered))}/{len(ordered)} commits",
            flush=True,
        )
    for row in manifest["seeds"]:
        if row["type"] == "userreport" and row["change"]:
            store.close_facts_by_issue(row["source_issue"], row["change"], "linked_fix")
    dag = Dag.from_commits(store.get_commits())
    closures = {}
    for row in store.all_fact_closures():
        closures.setdefault(row["fact_id"], set()).add(row["closing_commit"])
    projector = TimeLinearProjector(dag, store)
    for fid, commits in sorted(closures.items()):
        for commit in sorted(commits):
            projector.end_for(fid, commit, manifest["window"]["head"])
    with store._connect() as conn:
        facts = conn.execute("SELECT * FROM facts FOR ALL VALID_TIME").fetchall()
    by_id = {f["_id"]: f for f in facts}
    facts = [by_id[k] for k in sorted(by_id)]
    snapshots = {}
    with store._connect() as conn:
        for base in ordered:
            snapshots[base] = conn.execute(
                "SELECT * FROM facts FOR VALID_TIME AS OF %s",
                (dag.commits[base].committed_at,),
            ).fetchall()
    queries = [
        {
            "base": base,
            "seed_id": row["id"],
            "symbols": row["symbols"],
            "issue_refs": [row["source_issue"].lower()]
            if row.get("source_issue")
            else [],
        }
        for base in ordered
        for row in manifest["seeds"]
    ]
    manifest["replayed_commits"] = ordered
    for seed in manifest["seeds"]:
        if seed["construction"] == "natural":
            seed["truth_by_base"] = {
                base: check_seed(repo, base, **seed["evidence"]) for base in ordered
            }
    return dag, facts, closures, queries, snapshots


def export_live(store):
    with store._connect() as conn:
        runs = conn.execute("SELECT * FROM runs").fetchall()
        rounds = {
            r["round_id"] for r in runs if r.get("workflow_id", "").startswith("gh-")
        }
        rows = conn.execute(
            "SELECT *, _valid_from, _valid_to FROM facts FOR ALL VALID_TIME"
        ).fetchall()
    facts = {r["_id"]: r for r in rows if r.get("created_in_round") in rounds}
    return {
        "facts": [facts[k] for k in sorted(facts)],
        "closures": store.closures_for_facts(list(facts)),
    }
