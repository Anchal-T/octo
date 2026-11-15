---
date: 2026-11-15
tags: [agent-team, service, workflow]
aliases: [Agent Team Workflow Service]
---

# Workflow service: issue-scoped fact retrieval

Before each reproducer/fixer call, the existing journaled fact-query step now includes explicit issue references found in the current issue title/body: `#number` in the current repo, `owner/repo#number`, and GitHub issue URLs. It also includes the current issue when its identity is `gh-<repo_id>-<number>`. References are canonicalized and deduplicated, with at most 100 sent to Memory. Issue numbers are not inferred from prose or symbol names.

Memory applies the existing ancestry and expiry filters. Workflow retains confidence decay, the 15-fact bound and deterministic fact-read events. All `user_report` facts render inside the untrusted quoted block, even if a legacy row has taint below 2. Query results remain journaled, and empty results preserve existing prompts. No new configuration is introduced.

Related: [Memory](Memory.md), [locked P1 plan](../plans/P1-memory-wiring.md). HTTP routes are documented by [OpenAPI](http://localhost:8002/docs).

## Invalid Groq tool calls

The model client distinguishes HTTP 400 `tool_use_failed` responses whose message
reports a tool-call validation failure. The reproducer and fixer retry that same
turn at most twice on the pinned model. Other HTTP 400 responses remain terminal.
Each retry appends only a short corrective user message; failed generations are
never added to prompt history. The existing provider-error sanitizer redacts
credentials and bounds the stored body to 2 KiB; `ModelOutcome.provider_error_body`
preserves it, while `reason` holds the short correction.

Each retry has a distinct deterministic model step and request identity, plus a
journaled `tool_call_retry` event containing the run, turn, retry number, failed
request ID, pinned model, sanitized reason and body. Successful run records include
these event IDs. Replaying the workflow reuses the journaled calls and events.
Exhausting the two retries terminates the run; it does not become a quota wait.

These repo notes follow Anchal's read-only-vault instruction for this work.

Back to [[Agent Team - Overview]].
