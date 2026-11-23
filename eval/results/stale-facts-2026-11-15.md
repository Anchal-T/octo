# P1 stale-fact retrieval replay

Repo: Anchal-T/sqlparse. Window: 400 first-parent + 30 natural side commits.
Harness commit: `57ee9b0`. Seed: 1. Model requests: **0**.

| Condition | Queries | Injected | Stale | Stale usage | Points with stale | Expired withheld |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 55250 | 0 | 0 | n/a | 0 | 0 |
| C1 | 55250 | 39037 | 38763 | 99.30% | 32123 | 2579 |
| C2 | 55250 | 254 | 0 | 0.00% | 0 | 2579 |
| C3 | 55250 | 2837 | 2563 | 90.34% | 1903 | 2579 |

| Type | Logical seeds | Closures | Truly stale transitions | Precision | Recall | Delay median / p90 |
|---|---:|---:|---:|---:|---:|---|
| behavior | 20 | 20 | 20 | 1.0 | 1.0 | 0.0 / 0 |
| control | 5 | 0 | 0 | None | None | None / None |
| move | 20 | 20 | 20 | 1.0 | 1.0 | 0.0 / 0 |
| nonancestor | 20 | 0 | 0 | None | None | None / None |
| rename | 20 | 20 | 20 | 1.0 | 1.0 | 0.0 / 0 |
| revert | 20 | 20 | 20 | 1.0 | 1.0 | 0.0 / 0 |
| userreport | 20 | 10 | 10 | 1.0 | 1.0 | 0.0 / 0 |

| Threshold | Result | Value | Requirement |
|---|---|---:|---|
| T1 | PASS | 0.0 | C2 stale usage <= 5%  |
| T2 | PASS | 1.0 | relative cut >= 50%  |
| T3 | PASS | 1.0 | precision >= .8  |
| T4 | PASS | 1.0 | recall >= .7  |
| T5 | PASS | 1.0 | revert recovery = 100%  |
| T6 | FAIL | NOT RUN | live task success: no drop Live slice not run; no conclusion. |
| T7 | FAIL | NOT RUN | live tokens <= +20% Live slice not run; no conclusion. |
| T8 | PASS | 1.0 | user_report lifecycle / TTL = 100%  |

T6/T7 are marked FAIL only because the requested live slice was not run; they are unmeasured, not evidence of regression.

Revert checks: {'passed': 20, 'total': 20}. User-report checks: {'passed': 50, 'total': 50, 'ttl_isolated_reconfirmation_probes': 20, 'natural_ttl49_already_decayed': 10}.
Expired-but-true withdrawals: 2579. TTL is not counted as a closure.

## Coverage and limitations

- 430 ordinary history points are supplemented by 12 explicitly synthetic seed commits on local eval/p1-* branches. No branch was pushed.
- 120 logical seeds plus 5 stable controls; 20 revert reassertions make 145 physical fact rows. Natural seeds: {'rename': 1, 'move': 0}; remaining cases are controlled fixtures.
- Natural rename/move candidates are found independently by AST comparison, with ambiguous matches excluded; their named-definition evidence is checked at every query base. Missing cells use controlled fixtures. This is a wiring/seed-ground-truth evaluation, not evidence of generalization to arbitrary natural bugs.
- Queries are fixed canonical-symbol / source-issue groups for every seed at every replayed base. Historical identifiers remain queries after a rename, modeling stale references. The same query digest and fact pool are used in all four conditions; each block uses the production confidence/quoting/cap renderer.
- C1 deliberately permits facts from future and unmerged commits; C2 filters their ancestry. C3 is read from real XTDB valid-time AS OF snapshots after TimeLinearProjector; it has no ancestry test.
- Production Repo.replay indexes every selected commit, including merge-second-parent diffs. Ancestors outside the selected window are explicit metadata-only DAG support, never treated as indexed empty diffs.
- Expiry uses locked 0.8 confidence, 30-day half-life, 0.6 minimum and 50 main first-parent commits. Ten TTL-49 reports already decay in the natural history. Twenty separate re-confirmation probes isolate the hard TTL with the same locked parameters; these copies are never placed in the shared replay pool.
- Invalidation is scored over labeled (fact, change-commit) transitions. TTL is not a closure; non-ancestor leakage is a retrieval error rather than a missing closure. Reasserted revert facts and unchanged controls expose false closures.
- Live pool: 4 unlabeled facts, reported separately; up to 50 still need blind human labels. They are excluded from primary precision/recall and stale-rate claims.
- T6/T7 are intentionally unmeasured. No live slice and no model call was made. Local fixture branches and their SHA manifest are retained for inspection and deterministic rebuilding.

## Proposed live slice (not run)

Ten existing fork issue copies #5–#14, C0/C2 interleaved per issue, 20 runs on pinned qwen/qwen3.8-27b. The first two use 60cdc649726bf1bc4f1b336050560b336da715ec; the others use 09dbe1eec31ed1b3cd05366748865f11ba427d32. No upstream writes or merges.
The successful #10 smoke run used 19 completion HTTP requests (including four invalid-tool calls and one 429), versus 14 successful completions. Estimated central budget: **380 Groq completion requests**, plus one model-list GET; planning range **280–600**. At the observed 28,049 input + 1,075 output tokens/run, 20 similar runs would use about **582,480 tokens**, at least **73 minutes** at 8,000 TPM and about **3 daily 200k-token budgets**. Other issues may need more turns or fail early. These are estimates from one issue, not measured slice costs.
Before execution: add an explicit per-run C0/C2 selector (C0 skips retrieval), freeze a fact-pool snapshot so one condition cannot feed facts to the other, keep model retries counted, record resolve rate/tokens/latency/stale citations, and require approval to start. The live driver is not executed by this retrieval CLI.

## Execution audit

Two fresh isolated XTDB runs produced byte-identical JSON and seed manifests.
The [reproducibility receipt](stale-facts-2026-11-15-reproducibility.json) records
the SHA-256 digests. All 442 selected commits were indexed and replayed, with no
skipped commits or SCIP failures. No model credentials were present. Both
temporary databases/processes were removed; the existing volumes and stopped
xtdb-test container were preserved.

The fixed query set contains 55,250 blocks per condition: 50,000 on main, 3,750
on natural side commits, and 1,500 on controlled fixture commits. C2 injected
115 facts on main, zero on the 30 natural side points, and 139 on controlled
points. This sparse, intentionally stale-reference stress test should not be
interpreted as a natural-history stale-fact frequency or a live task-success
measurement. One of the 120 logical seeds is a usable natural nested-function
rename; the remaining cases are controlled constructions. Four real live-run
facts remain unlabeled and outside the scored metrics.

The ten TTL-49 examples had already expired from decay in the real history; the
shortest 50-commit span is about 136 days. Hard-TTL correctness was therefore
isolated using 20 separately counted re-confirmation probes with unchanged
0.8/30-day/0.6/50-commit defaults. There were 2,579 expired-but-still-true
query withdrawals in the shared pool.

The existing [fork PR #15 evidence comment](https://github.com/Anchal-T/sqlparse/pull/15#issuecomment-6893079367)
was refreshed from its recorded repro/fixer fact IDs and verified to contain
both lists, with the placeholder removed. No new PR or evaluation branch was
pushed. Local fixture branches remain in eval/.cache/stale-facts/sqlparse.

Before approving the proposed live slice, resolve the scheduling conflict:
the estimate exceeds one 200k-token daily budget, while the locked plan says
to stop if the pinned slice cannot finish within a day. No model was switched
and no live-slice invocation was started.

Final regression checks: **444 passed, 14 skipped, 0 failed**; opted-in XTDB memory suite: **108 passed, 0 skipped, 0 failed**. The tests include an actual ten-git-commit replay that reproduces the hand-computed metrics and temporal baseline.
