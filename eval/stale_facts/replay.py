"""Replay actual Memory invalidation and retrieval; never imports a model router."""

import math
from collections import Counter, defaultdict, deque
from datetime import datetime
from statistics import median

from agentteam_contracts import FactView

from services.memory.app.fact_query import filter_valid
from services.workflow.app.fact_block import render_fact_block

from .oracle import Oracle
from .report import thresholds


def _stats(closed, expected, oracle, facts):
    tp, delays, covered = 0, [], set()
    for fid, commit in closed:
        if oracle.truth(facts[fid], commit) != "stale":
            continue
        tp += 1
        for efid, start in expected:
            if efid != fid or start not in oracle.ancestors(commit):
                continue
            covered.add((efid, start))
            pending = deque([(commit, 0)])
            seen = set()
            while pending:
                node, dist = pending.popleft()
                if node == start:
                    delays.append(dist)
                    break
                if node not in seen:
                    seen.add(node)
                    pending.extend((p, dist + 1) for p in oracle.commits[node].parents)
    delays.sort()
    return {
        "closed": len(closed),
        "truly_stale": len(expected),
        "true_positive": tp,
        "precision": tp / len(closed) if closed else None,
        "recall": len(covered) / len(expected) if expected else None,
        "delay_median": median(delays) if delays else None,
        "delay_p90": delays[math.ceil(0.9 * len(delays)) - 1] if delays else None,
    }


def evaluate(dag, facts, closures, seeds, queries, time_snapshots):
    oracle = Oracle(dag, seeds)
    by_id = {f["_id"]: f for f in facts}
    metrics = {
        c: {
            "query_points": 0,
            "injected": 0,
            "stale_injected": 0,
            "points_with_stale": 0,
            "expired_withheld": 0,
        }
        for c in ["C0", "C1", "C2", "C3"]
    }
    kept = {}
    expired_true = 0
    points = defaultdict(lambda: defaultdict(Counter))
    fields = set(FactView.model_fields)
    for q in queries:
        base = q["base"]
        now = dag.commits[base].committed_at

        def matching(f, q=q):
            return any(a["symbol"] in q["symbols"] for a in f.get("anchors", [])) or (
                f.get("kind") == "user_report"
                and (f.get("source_issue") or "").lower() in q["issue_refs"]
            )

        candidates = [f for f in facts if matching(f)]
        for condition, row in metrics.items():
            row["query_points"] += 1
            source = (
                []
                if condition == "C0"
                else [f for f in time_snapshots[base] if matching(f)]
                if condition == "C3"
                else candidates
            )
            eligible, dropped = filter_valid(
                source,
                closures,
                dag,
                base,
                ancestry_filter=condition == "C2",
                base_time=now,
            )
            views = [
                FactView.model_validate(
                    {k: v for k, v in {**f, "id": f["_id"]}.items() if k in fields}
                )
                for f in eligible
            ]
            ids = render_fact_block(views, now).injected_ids
            if condition == "C2":
                kept[(base, q.get("seed_id"))] = ids
            stale = sum(oracle.truth(by_id[fid], base) == "stale" for fid in ids)
            row["injected"] += len(ids)
            row["stale_injected"] += stale
            row["points_with_stale"] += bool(stale)
            row["expired_withheld"] += dropped["expired"]
            points[base][condition].update(injected=len(ids), stale=stale, queries=1)
        for f in candidates:
            if oracle.truth(f, base) == "expired":
                seed = oracle.seeds[f["_id"]]
                expired_true += f["valid_from_commit"] in oracle.ancestors(
                    base
                ) and seed.get("change") not in oracle.ancestors(base)
    for row in metrics.values():
        row["stale_usage"] = (
            row["stale_injected"] / row["injected"] if row["injected"] else None
        )
        row["points_with_stale_share"] = (
            row["points_with_stale"] / row["query_points"]
            if row["query_points"]
            else None
        )
    expected = {(s["fact_ids"][0], s["change"]) for s in seeds if s.get("change")}
    closed = {(fid, c) for fid, cs in closures.items() if fid in by_id for c in cs}
    inv = _stats(closed, expected, oracle, by_id)
    types = {}
    for ty in sorted({s["type"] for s in seeds}):
        ids = {fid for s in seeds if s["type"] == ty for fid in s["fact_ids"]}
        types[ty] = {
            "n": sum(s["type"] == ty for s in seeds),
            **_stats(
                {x for x in closed if x[0] in ids},
                {x for x in expected if x[0] in ids},
                oracle,
                by_id,
            ),
        }
    revert = {"passed": 0, "total": 0}
    life = {
        "passed": 0,
        "total": 0,
        "ttl_isolated_reconfirmation_probes": 0,
        "natural_ttl49_already_decayed": 0,
    }
    for seed in seeds:
        if seed.get("restore"):
            revert["total"] += 1
            revert["passed"] += bool(
                set(kept.get((seed["restore"], seed["id"]), [])) & set(seed["fact_ids"])
            )
        for probe in seed.get("checks", []):
            life["total"] += 1
            if probe.get("isolate_ttl"):
                fact = dict(by_id[seed["fact_ids"][0]])
                # A controlled re-confirmation isolates the hard TTL from decay.
                # It is NOT substituted into the shared replay pool.
                fact["last_confirmed_at"] = dag.commits[probe["base"]].committed_at
                actual = bool(filter_valid([fact], closures, dag, probe["base"])[0])
                life["ttl_isolated_reconfirmation_probes"] += 1
                if (
                    probe["keep"]
                    and oracle.truth(by_id[fact["_id"]], probe["base"]) == "expired"
                ):
                    life["natural_ttl49_already_decayed"] += 1
            else:
                actual = bool(
                    set(kept.get((probe["base"], seed["id"]), []))
                    & set(seed["fact_ids"])
                )
            life["passed"] += actual == probe["keep"]
    return {
        "conditions": metrics,
        "invalidation": inv,
        "by_type": types,
        "revert_checks": revert,
        "lifecycle_checks": life,
        "expired_true": expired_true,
        "thresholds": thresholds(metrics, inv, revert, life),
        "points": {
            base: {c: dict(count) for c, count in rows.items()}
            for base, rows in sorted(points.items())
        },
    }


def unlabeled_summary(dag, exported, bases):
    """Live facts stay outside scored seeds until blind human labels exist."""
    facts = exported.get("facts", [])
    closures = defaultdict(set)
    for row in exported.get("closures", []):
        closures[row["fact_id"]].add(row["closing_commit"])
    known = [f for f in facts if f["valid_from_commit"] in dag.commits]
    totals = {c: 0 for c in ("C0", "C1", "C2", "C3")}
    for base in bases:
        now = dag.commits[base].committed_at
        for c in totals:
            rows = [] if c == "C0" else known
            if c == "C3":
                rows = [
                    f
                    for f in rows
                    if datetime.fromisoformat(f["last_confirmed_at"]) <= now
                ]
                # Live pool is exported read-only; projected temporal bounds are
                # exported too, rather than guessing C3 using DAG ancestry.
                rows = [
                    f
                    for f in rows
                    if not f.get("_valid_to")
                    or now < datetime.fromisoformat(f["_valid_to"])
                ]
            eligible, _ = filter_valid(
                rows, closures, dag, base, ancestry_filter=c == "C2", base_time=now
            )
            views = [
                FactView.model_validate(
                    {
                        k: v
                        for k, v in {**f, "id": f["_id"]}.items()
                        if k in FactView.model_fields
                    }
                )
                for f in eligible
            ]
            totals[c] += len(render_fact_block(views, now).injected_ids)
    return {
        "facts": len(facts),
        "known_origins": len(known),
        "human_labels": 0,
        "scored_in_primary_metrics": False,
        "injected_by_condition": totals,
        "fact_ids": [f["_id"] for f in facts],
    }


def main():
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path

    from agentteam_common import Settings

    from services.memory.app.store import MemoryStore

    from .backend import execute
    from .report import markdown
    from .seed import construct, window

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--side-branches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("seeds.json")
    )
    parser.add_argument("--harness-commit", default="working-tree")
    parser.add_argument("--live-pool", type=Path)
    args = parser.parse_args()
    if not 300 <= args.window <= 500:
        parser.error("the locked P1 window must be 300–500")
    if not 1 <= args.seed <= 98:
        parser.error("seed must fit two-digit local branch names")
    if args.side_branches < 0:
        parser.error("--side-branches must be nonnegative")
    store = MemoryStore(Settings.from_env())
    if not store.scip_indexer:
        parser.error(
            "SCIP_REPO_PATH, SCIP_CACHE_DIR and SCIP_PROJECT_NAME are required"
        )
    if Path(store.settings.scip_repo_path).resolve() != args.repo.resolve():
        parser.error("--repo must be the writable SCIP_REPO_PATH clone")
    with store._connect() as conn:
        existing = conn.execute("SELECT _id FROM facts LIMIT 1").fetchall()
    if existing:
        parser.error(
            "use a fresh isolated XTDB; refusing to mix pools or modify live facts"
        )
    manifest = construct(
        args.repo, window(args.repo, args.window, args.side_branches), args.seed
    )
    progress = lambda *a, **kw: print(*a, file=sys.stderr, **kw)
    dag, facts, closures, queries, snapshots = execute(
        store, args.repo, manifest, progress
    )
    result = evaluate(dag, facts, closures, manifest["seeds"], queries, snapshots)
    result.update(
        seed=args.seed,
        window=manifest["window"],
        harness_commit=args.harness_commit,
        memory_code_sha256=hashlib.sha256(
            Path(sys.modules[MemoryStore.__module__].__file__).read_bytes()
        ).hexdigest(),
        query_digest=hashlib.sha256(
            json.dumps(queries, sort_keys=True).encode()
        ).hexdigest(),
        seeded_fact_rows=len(facts),
        logical_seeds=sum(s["type"] != "control" for s in manifest["seeds"]),
        negative_controls=sum(s["type"] == "control" for s in manifest["seeds"]),
        model_requests=0,
        synthetic_commits=manifest["synthetic_commits"],
        scip_failures=[],
        skipped_commits=[],
        live_pool=unlabeled_summary(
            dag,
            json.loads(args.live_pool.read_text()) if args.live_pool else {},
            manifest["replayed_commits"],
        ),
    )
    result["notes"] = [
        f"{len(manifest['window']['main']) + len(manifest['window']['side'])} ordinary history points are supplemented by {len(manifest['synthetic_commits'])} explicitly synthetic seed commits on local eval/p1-* branches. No branch was pushed.",
        f"120 logical seeds plus {result['negative_controls']} stable controls; 20 revert reassertions make {len(facts)} physical fact rows. Natural seeds: {manifest['natural_seed_counts']}; remaining cases are controlled fixtures.",
        "Natural rename/move candidates are found independently by AST comparison, with ambiguous matches excluded; their named-definition evidence is checked at every query base. Missing cells use controlled fixtures. This is a wiring/seed-ground-truth evaluation, not evidence of generalization to arbitrary natural bugs.",
        "Queries are fixed canonical-symbol / source-issue groups for every seed at every replayed base. Historical identifiers remain queries after a rename, modeling stale references. The same query digest and fact pool are used in all four conditions; each block uses the production confidence/quoting/cap renderer.",
        "C1 deliberately permits facts from future and unmerged commits; C2 filters their ancestry. C3 is read from real XTDB valid-time AS OF snapshots after TimeLinearProjector; it has no ancestry test.",
        "Production Repo.replay indexes every selected commit, including merge-second-parent diffs. Ancestors outside the selected window are explicit metadata-only DAG support, never treated as indexed empty diffs.",
        "Expiry uses locked 0.8 confidence, 30-day half-life, 0.6 minimum and 50 main first-parent commits. Ten TTL-49 reports already decay in the natural history. Twenty separate re-confirmation probes isolate the hard TTL with the same locked parameters; these copies are never placed in the shared replay pool.",
        "Invalidation is scored over labeled (fact, change-commit) transitions. TTL is not a closure; non-ancestor leakage is a retrieval error rather than a missing closure. Reasserted revert facts and unchanged controls expose false closures.",
        f"Live pool: {result['live_pool']['facts']} unlabeled facts, reported separately; up to 50 still need blind human labels. They are excluded from primary precision/recall and stale-rate claims.",
        "T6/T7 are intentionally unmeasured. No live slice and no model call was made. Local fixture branches and their SHA manifest are retained for inspection and deterministic rebuilding.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    args.manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "thresholds": result["thresholds"],
                "replayed_commits": len(manifest["replayed_commits"]),
                "model_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
