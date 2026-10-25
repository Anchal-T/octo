# Agent Team — instructions for coding agents

This folder is the code for **Agent Team**: a team of LLM agents that maintains a
software product on its own. Product signals (issues, user feedback, telemetry)
come in, agents turn them into fixes through a **repro → fix → test → PR**
pipeline, coordinate in **BSP rounds**, and earn autonomy per product area
through a **trust ladder**. Underneath is a memory layer that records what the
team knew, when, and why, with facts tied to git commits.

It is also a research project. The core question is whether **commit-DAG-anchored
facts** (facts valid on a commit's ancestry, closed instead of deleted when the
code changes) cut stale-fact errors compared with time-linear memory. The
evaluation replays a real repo's history.

Nothing here is built yet. Week 1 starts 2026-10-26.

---

## 1. Read the notes first

The design lives in Anchal's Obsidian vault, not in this repo. Read the relevant
note **before** you design, write code, or ask a question the notes already answer.

Vault: `~/anchal/Obsidian/General/Agent Team/`

| Note | Read it when |
|---|---|
| `Agent Team - Overview.md` | Always, first. The hub and the idea in short. |
| `Design Doc.md` | Architecture, goals/non-goals, eval plan, milestones, risks (the HLD). |
| `Low Level Design.md` | Writing code: package layout, dependency rules, classes, tables, sequences, idempotency, error taxonomy, `agentteam.yaml` schema, **build order (§7.2)**. |
| `MVP and UX Blueprint.md` | Anything user-facing: CLI, GitHub PR evidence comment, run states, copy. Its dashboard screens are not decided yet. |
| `Agent Team - Data Model (XTDB).md` | XTDB tables and tested SQL queries. |
| `Agent Team - v1 Loop and Evaluation.md` | Agent roles, Restate orchestration, BSP round, fact invalidation via SCIP, history-replay eval. |
| `Agent Team - Database Choice.md` | Why XTDB v2 + LanceDB (don't reopen without a reason). |
| `Agent Team - Data Architecture Research.md` | Background research, papers, gaps. |
| `Agent Team - Brainstorm.md` | Original problems and patterns. |

Other notes in the vault (system design: consistent hashing, key-value store,
networking, etc.) can be read when a topic needs them. Search with `rg` over
`~/anchal/Obsidian/General`, or use the Obsidian CLI (`~/.local/bin/obsidian`,
needs the Obsidian app open).

**Do not read or quote** the private Incident Report journal in the vault.

The notes are the source of truth for design. If code and notes disagree, say so
and ask; don't silently pick one.

### 1.1 Project documentation lives in Obsidian (decided 2026-10-25)

There is **no `docs/` folder in this repo**. All project documentation goes in the
vault under `~/anchal/Obsidian/General/Agent Team/Docs/`:

- one note per service (what it owns, its API, its tables, its config, how to run it)
- architecture, data model, and run/deploy notes
- `Decisions/`: one short note per decision (what, why, date, what it replaces)

Rules:
- When you change a service's API, tables or config, update its doc note in the
  same piece of work.
- Write in the vault's style: YAML front matter (`date`, `tags`, `aliases`),
  `[[wikilinks]]` to related notes, and a "Back to [[Agent Team - Overview]]" line.
- Don't edit the existing design notes (Overview, Design Doc, Low Level Design,
  MVP and UX Blueprint, Data Model, v1 Loop, research notes) unless Anchal asks.
- FastAPI's generated OpenAPI is the endpoint reference; doc notes link to it
  instead of hand-copying every endpoint.
- The vault exists only on Anchal's laptop (backed up to `Anchal-T/General-obsidian`).
  An agent without access to it should say so, not create docs in the repo.

## 2. Decisions that are newer than the notes

Some notes predate these. These win:

- **No sqlglot** (decided 2026-10-25). Don't wire sqlglot in. Replacements:
  - **Live demo repo: `andialbrecht/sqlparse`** (a fork of it; BSD-3, ~4.4k
    lines of source, pytest suite ~1.4s, has merge commits). Live PRs go to the
    fork only, never upstream.
  - **Eval repo: `marshmallow-code/marshmallow`** (MIT, ~5.2k lines of source,
    pytest ~1.4s, 720 merges for history replay, 72 real SWE-rebench tasks
    incl. the filtered split). This replaces the SWE-rebench sqlglot tasks.
  - The sqlglot-specific parts of the notes (task loader, area list such as
    dialects/parser/optimizer) must be re-targeted to these repos; ask before
    choosing new area lists.
  - scip-python indexing has not been tested on either repo yet; check it first.
- **Build in phases.** Each phase does its own job end to end before the next one
  is added. Don't scaffold the whole system at once.
- **Approval** = GitHub PR review, with a CLI fallback. Whether there is a
  dashboard at all is undecided; if one is built later it has no approve button.
- **GitHub events** arrive by polling, not webhooks.
- **Parked/resumed run state** is read from Restate's journal, not a separate table.
- **Models are used through APIs** (decided 2026-10-25), not self-hosted as the main path.
  Starting candidates per role (not final; Anchal decides later; strictly free, no paid credit):

  | Role | Model | Provider |
  |---|---|---|
  | fixer | `z-ai/glm-5.3` | NVIDIA NIM (free endpoint) |
  | reproducer | `moonshotai/kimi-k2.6` (alt: `deepseek-ai/deepseek-v4.1-flash`) | NVIDIA NIM |
  | tester-reviewer | `gemini-3.5-flash-lite` | Gemini API free tier |
  | curator | Gemma 4 31B | Gemini API free tier |
  | triager | `openai/gpt-oss-20b` (short prompts only, 8K TPM) | Groq free plan |
  | spare | `thinkingmachines/inkling-small:free` | OpenRouter (50 req/day shared cap) |

  Rules: pin one model per role during eval (no silent fallback; turn off OpenRouter
  provider fallback); retry 429s with backoff on the same model and count retries;
  record model ids and dates per condition; never put secrets or `.env` contents in a
  prompt (free endpoints log/train on prompts). NVIDIA's trial terms say "evaluation
  and prototyping"; check them before using NIM for live PRs.
- **Free models only in the initial phase**: free API tiers (and free GPUs if ever
  needed). Paid APIs may be considered later, only when Anchal says so. Keep the
  model layer provider-agnostic so switching is a config change.
- **Commit-DAG validity** has been prototyped: facts carry `valid_from_commit`, a
  `fact_closures` table has one row per closing commit, the ancestor set is
  computed in Python, and merges also diff against their second parent. XTDB
  valid time on `main` is the time-linear baseline.

## 3. Architecture: microservices (decided 2026-10-25)

Agent Team is built as **FastAPI microservices in one monorepo**, so it stays
readable as features grow and pieces can be deployed separately. This replaces
the layer-based package layout in `Low Level Design.md` §1; the LLD's classes,
tables, sequences and contracts still apply, they just live inside services.

### 3.1 Services

| Service | Owns | Phase |
|---|---|---|
| `github` | GitHub polling, PR evidence comment, commit status, PR review → approval; its polling cursor | 1 |
| `workflow` | Restate handlers: issue run (triage → repro → fix → test → PR), round, area. Runs the Docker sandbox at first | 1 |
| `memory` | **Single writer** of facts, closures, proposals, rounds (XTDB) and vectors (LanceDB); the BSP commit; retrieval. Runs code indexing (SCIP, symbol diff, `on_push`) at first | 1 |
| `models` | Model router, quota, response cache in front of `llama-server` (GPU/Kaggle) and free API tiers | 1 |
| `code-index` | Split out of `memory` later | later |
| `sandbox` | Split out of `workflow` later | later |
| `intake` | PII scrub, taint, its own PII vault table; when signals beyond GitHub issues arrive | later |
| `gateway` | **Dashboard is undecided** (Anchal will decide later). Reserved name only; don't build it | TBD |

Start with the four phase-1 services. Split out the others only when their phase
needs them; each already has a reserved name here.

### 3.2 Data ownership

- **`memory` is the only service that writes facts, closures, proposals and
  rounds.** A BSP round must commit as **one local XTDB transaction** inside
  `memory`. Other services send proposals through `memory`'s API. Never add a
  second writer to these tables and never split a round across services.
- Small, isolated state may be owned elsewhere (e.g. `intake`'s PII vault,
  `github`'s polling cursor) because nothing commits together with it.
- Other services may read through `memory`'s API (or read-only views if
  needed), never write.

### 3.3 Communication

- **Restate durable calls** for anything inside a workflow step. This is what
  makes crash/resume work without repeating model calls.
- **Plain FastAPI HTTP** for reads and anything outside a workflow (CLI,
  GitHub polling).
- Each service is a FastAPI app; its Restate handlers are mounted alongside
  (verify the current Restate Python SDK docs before wiring this).
- The LLD's Restate rules still hold: journaled bodies are plain functions that
  never touch the context, and handler code outside `ctx.run` is deterministic.

### 3.4 Repo layout

```text
agent-team/
  packages/
    contracts/     # pydantic request/response models shared by all services
    common/        # ids, errors, config, logging, XTDB client
  services/
    <name>/
      app/         # main.py (FastAPI), api/, handlers/ (Restate), core/, store/
      tests/
      Dockerfile
      pyproject.toml
  eval/            # offline replay and metrics; calls services over HTTP
  deploy/
    compose.yaml   # XTDB, Restate server and all services for local dev
  AGENTS.md
```

- Services share **only** `packages/contracts` and `packages/common`, never
  each other's code. Enforce with `import-linter`.
- A service's HTTP and Restate API is its only public surface.

### 3.5 Stack

- Python, FastAPI, Restate (Python SDK)
- XTDB v2 (bitemporal SQL) as the main store, LanceDB for vectors
- Embeddings: **Cloudflare Workers AI `@cf/qwen/qwen3-embedding-0.6b` is the main
  embedder** (decided 2026-10-25). Local Qwen3-Embedding-0.6B is only a fallback
  (offline or quota). This reverses the LLD's local-first `embedder.py`. Keep the
  same model on both paths so vectors stay comparable.
- SCIP for the code index and fact anchors
- GitHub REST for issues, PRs, reviews, commit status
- Docker / docker compose for local runs

Check XTDB 2.x constraints in `Design Doc.md` §10.1 before writing SQL.
Use current library docs (e.g. Context7) instead of memory for Restate, FastAPI and XTDB APIs.

## 4. How to work

- Follow the service layout and data-ownership rules in §3.
- Tests first where it makes sense (`tdd` skill). Every side effect gets an
  idempotency key (LLD §5.2). A crash must resume without repeating model calls.
- Never invent metrics, results, SHAs or issue titles. Example data must be
  labelled as example data.
- Keep secrets in env vars or `.env` (git-ignored). Never commit or print them.
- Don't push, open PRs, or touch other repos unless Anchal asks.

## 5. Skills and other resources

- Shared agent skills: `~/.agents/skills` (e.g. `tdd`, `diagnosing-bugs`,
  `code-review`, `codebase-design`, `grill-with-docs`, `karpathy-guidelines`).
- Figma: FigJam flows board `bOyeke2Fg35WY3NFk3t5u0`, wireframes
  `2y6FqovMEHjkB2r2etnFih` (links in `MVP and UX Blueprint.md`).
- College GPU (for local model serving/evals): `ssh -p 2222 anchal@10.0.24.7`
  (4× K80, shared; be polite with resources).
- Related past project for patterns: `~/anchal/code/CodeFury` (multi-agent coding
  orchestrator with git worktrees and LangGraph checkpoints).

## 6. Open questions

Listed in `Design Doc.md` §10.3 and `Low Level Design.md` §8. Next topics to settle
were models and docs. Ask Anchal before deciding any of them.
