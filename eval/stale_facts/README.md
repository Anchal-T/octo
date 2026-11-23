# P1 retrieval-only replay

Run as a Memory-owned maintenance process against a **fresh isolated XTDB**.
The harness never starts Workflow, GitHub polling or a model client. Proposals
use MemoryStore's normal BSP transaction, invalidation uses Repo.replay and
SCIP project `sqlparse` version `0`, and C3 reads actual XTDB valid-time AS OF
snapshots after TimeLinearProjector. Harness SQL is read-only.

Use a writable clone of `repos/sqlparse`; set SCIP_REPO_PATH to it,
SCIP_PROJECT_NAME=sqlparse, SCIP_CACHE_DIR to the existing SCIP cache, and
XTDB_HOST/PORT to an isolated instance. No API secrets are needed. In the memory
image use `/app/.venv/bin/python`.

```
uv run python -m eval.stale_facts.replay --repo <writable-clone> \
  --window 400 --side-branches 30 --seed 1 \
  --harness-commit <code-sha> --live-pool <optional-read-only-export.json> \
  --out eval/results/stale-facts-2026-11-15.json
uv run python -m eval.stale_facts.report eval/results/stale-facts-2026-11-15.json
```

The seed table is saved to `eval/stale_facts/seeds.json`. Rebuild using the same
repo, seed and code with a fresh isolated XTDB to reproduce the JSON. Existing
facts cause an explicit refusal. Branches use fixed author/dates and git
objects/refs, leaving the checkout untouched. The actual run used a separate
local clone at `eval/.cache/stale-facts/sqlparse`. No remote push is part of the
CLI. Retain the branches and manifest for inspection.

430 ordinary history commits are supplemented by 12 controlled fixture commits
for all six stale types. Natural rename/move candidates are found by independent
AST comparison, with ambiguous matches excluded. Twenty logical seeds per type,
five stable controls and 20 revert reassertions give 145 physical facts.
Synthetic claims and source issues are labeled examples. Their scores establish
wiring and validity behavior, not generalization to arbitrary natural bugs.

Each seed's canonical symbols/source issue define the same query at every base.
Queries intentionally retain obsolete names to model stale references. All four
conditions see the same pool and queries, using the production renderer with
decay, 15-fact cap and untrusted quoting. C1 intentionally permits future and
unmerged facts; C2 filters their ancestry. An independent oracle checks actual
named-definition evidence for natural seeds and known fixture transitions.
Unlabeled live facts stay outside primary metrics pending blind human labels.

User reports keep the locked 0.8 confidence, 30-day half-life, 0.6 minimum and
50-main-commit TTL. Natural decay is scored normally. Separate, counted
re-confirmation copies isolate the TTL boundary at 49/50 without changing the
shared pool or those parameters.

T6/T7 are unmeasured until the live slice is approved: FAIL with measured=false
means not run, not a claimed regression. The report includes the proposed slice
and cost estimate; this CLI cannot execute it.
