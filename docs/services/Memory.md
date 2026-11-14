---
date: 2026-11-15
tags: [agent-team, service, memory]
aliases: [Agent Team Memory Service]
---

# Memory service: issue-scoped fact retrieval

Step 5b extends the existing fact query with optional `issue_refs`, a list of up to 100 `owner/repo#number` references. The generated endpoint reference is [OpenAPI](http://localhost:8001/docs).

The query combines SCIP symbol matches with `user_report` facts whose `source_issue` matches one of those references. Repository matching is case-insensitive. Reports need no anchors; unrelated reports and other fact kinds are not selected by issue reference. Duplicate facts are removed before applying the existing ancestry, closure, decay, first-parent TTL and limit rules. Omitting `issue_refs` retains symbol-only retrieval. No new table or configuration is introduced.

Related: [Workflow](Workflow.md), [locked P1 plan](../plans/P1-memory-wiring.md).

Back to [[Agent Team - Overview]].


## R2 history ingestion repair

Git diff bodies can contain non-UTF-8 fixture data in historical sqlparse commits. The hunk parser needs ASCII headers only, so diff body decoding replaces invalid bytes. Paths still come from the separate NUL-delimited byte stream and retain their existing decoding. A regression test covers a Python change alongside legacy-encoded fixture text.
