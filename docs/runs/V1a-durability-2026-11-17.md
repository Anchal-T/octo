# V1a durability proof — 2026-11-17 (IST)

**PASS — isolated test proof.** Production IssueRunV2, service configuration, model pins and the scheduled P1 resume are unchanged. No real provider, Groq or GitHub call, vault edit, push or production deployment was performed for this proof.

## Reproduce

```bash
AGENTTEAM_TEST_DURABILITY=1 \
AGENTTEAM_DURABILITY_RECEIPT=/tmp/v1a-receipt.json \
uv run pytest services/workflow/tests/test_durability_integration.py -q
```

Requires Docker and the pinned Restate image `docker.restate.dev/restatedev/restate:1.8.1`; Python SDK **1.1.0**. The integration uses unique containers, a dedicated named data volume and dynamically allocated host ports. The Docker volume avoids Docker Desktop's restrictions on pytest's `/tmp` bind mounts. Container, volume and subprocess cleanup run on failure too. No production ports or volumes are used.

## Observed evidence

[Machine-readable receipt](../../eval/results/v1a-durability-2026-11-17.json).

| Check | Observed result |
|---|---|
| Two journaled fake-model turns | **2 actual body executions**, one for each stable request ID; 2 saved completions |
| Timer wait | Worker absent; in-flight requests 0; fake calls 1 |
| Restart Restate during timer | Persisted deployment/journal retained; timer resumed automatically |
| Timer duration | 20 seconds requested; **20.541 seconds** between observed turns, within the declared startup tolerance |
| Approval wait | Worker absent; fake calls remain 2 |
| Restart Restate during approval | Revision and waiting stage retained; no repeated turn |
| Stale revision | `stale_revision`, stays waiting |
| Correct approval | Worker verified absent immediately beforehand, then automatically started; `recorded`, workflow completes |
| Duplicate conflicting decision | `already_decided`; original decision unchanged |
| Cleanup | No worker alive; test container/volume removed |

A separate fast test deliberately executes the same fake body twice: execution count becomes 2 while saved completions remain 1. Thus a response cache cannot hide a repeated body in the end-to-end assertions. A worker-start failure regression was observed red (`in_flight=1`), then fixed and verified green (`in_flight=0`).

## Validation

- Normal full suite: **469 passed, 15 skipped**. The real durability test is intentionally opt-in; the remaining skips are inherited XTDB opt-ins.
- Explicit real durability test: **1 passed**; focused probe tests: **3 passed**.
- Repository-wide Ruff and formatting of all four new Python files: **pass**.
- Independent review: cleanup ordering and explicit cold approval activation were fixed; **no remaining blocking findings**.

## Implementation boundaries

All four modules live under `services/workflow/tests/`: versioned `DurabilityProbeV1`, the fake provider/counter, a request-response FastAPI worker, a local activation proxy and one end-to-end integration test. Existing production workflow APIs and journals are not changed, so no vault service API/config note needs editing.

The worker reaches zero processes between requests; **the activator/control-plane listener and Restate remain available**. This is a tested local activation pattern, not a production serverless rollout. The fixture's binding to the Docker host interface exists only during the test, uses synthetic data and carries no credentials. The test worker receives a minimal environment.

This proves replay of acknowledged journal steps across worker exits and Restate restarts. The uncertain remote-provider acknowledgement window, global model authority, dead-letter policy and stateful app snapshots remain V1b/V1c work. No strict exactly-once claim about a real Groq HTTP request is inferred from this fake-provider proof.
