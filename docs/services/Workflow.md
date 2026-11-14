---
date: 2026-11-15
tags: [agent-team, service, workflow]
aliases: [Agent Team Workflow Service]
---

# Workflow service: issue-scoped fact retrieval

Before each reproducer/fixer call, the existing journaled fact-query step now includes explicit issue references found in the current issue title/body: `#number` in the current repo, `owner/repo#number`, and GitHub issue URLs. It also includes the current issue when its identity is `gh-<repo_id>-<number>`. References are canonicalized and deduplicated, with at most 100 sent to Memory. Issue numbers are not inferred from prose or symbol names.

Memory applies the existing ancestry and expiry filters. Workflow retains confidence decay, the 15-fact bound and deterministic fact-read events. All `user_report` facts render inside the untrusted quoted block, even if a legacy row has taint below 2. Query results remain journaled, and empty results preserve existing prompts. No new configuration is introduced.

Related: [Memory](Memory.md), [locked P1 plan](../plans/P1-memory-wiring.md). HTTP routes are documented by [OpenAPI](http://localhost:8002/docs).

These repo notes follow Anchal's read-only-vault instruction for this work.

Back to [[Agent Team - Overview]].
