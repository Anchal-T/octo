# P1 close-out: independent Standards and Spec review

Model: `opencode/hy4-flash-free`, two independent read-only OpenCode sessions. Fixed baseline: `d7ffb00` (P1), plus working changes since `ea87ecf` (close-out). The parent independently verified the final full suite (466 passed, 14 skipped), isolated XTDB suite (108 passed) and repository-wide Ruff. Reviewer references to no model calls mean no octo/Groq evaluation calls, not the HY4 review inference itself.

## Independent STANDARDS review — P1 close-out

**Verdict: no blocking findings.** Against source (not agent conclusions), I found no correctness or standards breach in the close-out that should block the parent's finalization. Receipts/narrative accurately state the known limits (1726 tokens left, empty C2 snapshots, 4 terminal/1 skip/5 pending, T6/T7 unmeasured); I make no comparison claim and requested no new spend, tests, or writes.

### Documented standards — verified, no breach
- **Single writer / data ownership.** `memory` alone writes facts/closures/proposals/rounds (`store.py`); `github` and `workflow` reach it only over HTTP (`memory_client.py`, `issue_run_v2.py`). No cross-service app imports exist under `services/`; only `eval` imports app modules (LLD §1 permits this, non-handler modules only).
- **Idempotency (LLD §5.2/§3.2).** `stable_id` keys + `request_sha256` + `ASSERT` guards in `_write_once` and one-transaction `commit_round`; closure/commit ids deterministic. Durable side effects in `issue_run_v2.py` are all journaled `ctx.run_typed` with idempotency keys.
- **Secret handling.** Redaction in `client.py:32–37,1020–1031`; `live.py` runs ledger/pin/`/models` checks via compose `exec` and returns sanitized JSON, never printing keys; `gh api` reads are authenticated but not echoed. No secret/env prints found.
- **Pins/cap/frozen inputs.** `verify_live_config` (`live.py:329–337`) enforces all four roles `qwen/qwen3.8-27b` and `GROQ_TPD==190000`; `_check_run_models` (`:546–551`) rejects any reported-model drift; C2 snapshot is frozen into the checkpoint (`:411–434`). No fallback/switch path exists in `ModelClient`/`pin_for`. Consistent with the receipt.
- **XTDB constraints (Design Doc §10.1).** Parenthesized `(a.anc).symbol`, `EXISTS(UNNEST(...))`, no `SELECT` inside write transactions, parameterized temporal bounds — all honored.
- **Close-out logic.** Quota/throttle/`unknown` now pause without advancing the two-failure streak; a bare 500 is classified `unknown`/harness (`live.py:219–275,728–739`), which is conservative and matches the narrative. Tests at `test_live.py` protect each change.

### Potential smells (non-blocking; no refactor demanded)
1. **Duplicated Code** — `eval/tests/test_live.py:234–348` and `:429–539` repeat ~90 lines of `FakeResponse`/`FakeClient`/`fake_usage`/checkpoint setup. A single small helper or pytest fixture would remove ~80 lines; no new framework needed. Pure test ergonomics, no correctness risk.
2. **Documented-standard gap** — AGENTS.md §3.4 says enforce service boundaries with `import-linter`, but no config/CI exists. Pre-existing and unrelated to close-out; boundaries are currently respected by convention.

Human merge/push approval remains separate.

## Independent SPEC Review — P1 close-out (read-only)

**Verdict: No findings.** The close-out working-tree diff (`git diff ea87ecf` on `eval/stale_facts/live.py`, `eval/tests/test_live.py`, `docs/plans/P1-memory-wiring.md`) and the checked-in P1 source satisfy the authoritative scope (P1-memory-wiring D5/D11/§5.6; octo-v2 V0 §379–394 and D1 §636). Review was against source; no edits, tests, model calls, or provider operations were performed.

Requirement-by-requirement (all verified):

- **Quota/throttle ≠ provider terminal failure.** `is_provider_failure()` (live.py:243–247) requires `status=="failed"` plus a real marker; `should_pause_after_result()` (:219–228) pauses quota/throttle/unknown. `provider_failure_streak()` (:250–260) breaks on them. Tests `test_quota_and_throttle_pause…`, `test_real_provider_failure_markers_count_consecutively`.
- **Bare 500 → unknown/harness inspection.** `terminal_failure_result(500)` (live.py:263–275) returns `status="unknown"`, `failure_source="harness"`; not provider-attributed; pauses via `should_pause_after_result("unknown")` with "stopping for inspection, no later slot started" (:728–739). Test `test_generic_http_500_is_unknown_harness…`.
- **Two-failure stop.** `provider_failure_streak>=2` (:623–628, :722–727) counts trailing *provider-marked* outcomes only; task misses reset it. No auto-count of raw HTTP errors.
- **Distinct classes.** `_write_daily_status()` (:492–502) separates quota/throttle pauses, unknown/harness stops, and other terminal failures; receipt per-slot `failure_reason` attributes each.
- **No next slot after quota.** `break` at :728–739; test `test_quota_pause_does_not_launch_later_slots` asserts only one GET.
- **No false "10 executed".** `plan_counts()` (:189–193) and final summary (:783–789) report completed/skipped/pending; skip #6/C2 preserved.
- **Stable checkpoint/results.** Checkpoint sha256 `76befbe3…` matches receipt; 4 completed / 1 skip / 5 pending; `load_checkpoint()` rejects order/pin drift (:285–291, :566–567); retired attempt 90 and old counters intact.
- **Budget hardcap.** `DAILY_ESTIMATED_TOKEN_LIMIT=190000` (:22); `verify_live_config()` enforces `GROQ_TPD==190000` (:333–336); four pins `qwen/qwen3.8-27b`; preflight reservation 6,256 cannot fit remaining 1,726 → pause. No switch, push, vault edit, or secret exposure found.

**Ledger arithmetic validated against sanitized receipt:** 51 completed + 12 throttled = 63 counted requests; 144,054 + 44,220 = 188,274 reserved; retries 8; all-state 76/238,733/12; slots 59/179,793/127,679 + retired 4/8,481/3,041 = daily 63/188,274/130,720. All consistent.

**Protocol/leak flag:** Close-out changed no request body (`_make_request` untouched); mid-slice text-action/accounting changes are documented, not silently pooled. Frozen C2 snapshot is built once pre-run (`prepare_checkpoint`, :396–434) and consumed for both repro and fix (`query_fact_block`, issue_run_v2.py:102–114; C2 requires snapshot, C0 forbids it, models.py:356–362). Run-written facts cannot leak into the empty pool.

Per known limits: all C2 snapshots empty and T6/T7 remain unmeasured/inconclusive; no additional live spending or comparison claim is warranted. Human merge/push approval remains separate.

Standards: 0 blocking findings, 2 non-blocking observations (duplicated test setup and existing lack of import-linter enforcement). Spec: 0 findings. Human approval, merge and push remain separate.
