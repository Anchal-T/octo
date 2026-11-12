from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq
from agentteam_common import Settings, load_agentteam_config
from agentteam_common.areas import areas_for_diff
from agentteam_contracts import ModelCallRequest, ModelOutcome, Sampling
from dotenv import load_dotenv

from services.workflow.app.models import Ledger, ModelClient, Store
from services.workflow.app.models.pools import pin_for
from services.workflow.app.triage import parse_triage, triage_messages

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = "nebius/SWE-rebench"
DEFAULT_REPO = "marshmallow-code/marshmallow"
Verdict = Literal["hit", "miss", "parse_failure", "no_gold_area", "call_failed"]

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_DIFF_AB_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")


@dataclass(frozen=True)
class TaskItem:
    instance_id: str
    repo: str
    problem_statement: str
    patch: str
    base_commit: str | None = None
    created_at: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    revision: str
    repo: str
    tasks: tuple[TaskItem, ...]
    split_counts: dict[str, int]
    filtered_subset_of_test: bool | None = None


@dataclass(frozen=True)
class TaskOutcomeRecord:
    instance_id: str
    changed_paths: tuple[str, ...]
    gold_areas: tuple[str, ...]
    call_outcome: str
    predicted_area: str | None
    predicted_kind: str | None
    predicted_priority: str | None
    parse_failure: bool
    verdict: Verdict
    model: str | None
    tokens_in: int
    tokens_out: int
    retry_count: int
    error_code: str | None = None
    http_status: int | None = None


@dataclass(frozen=True)
class AreaStats:
    area: str
    gold_support: int
    predicted_scorable: int
    predicted_no_gold: int
    direct_hits: int
    gold_task_hits: int
    gold_misses: int
    false_positives: int


@dataclass(frozen=True)
class EvalSummary:
    dataset_id: str
    dataset_revision: str
    repo: str
    model_pinned: str
    provider: str
    evaluated_at: str
    split_counts: dict[str, int]
    filtered_subset_of_test: bool | None
    total_tasks: int
    attempted: int
    scorable: int
    hits: int
    misses: int
    parse_failures: int
    no_gold_area: int
    failed_calls: int
    multi_area_gold_count: int
    accuracy_scorable: float | None
    accuracy_all_attempted: float | None
    area_stats: tuple[AreaStats, ...]
    records: tuple[TaskOutcomeRecord, ...]
    blocker: str | None = None
    layout_diagnostic: dict[str, Any] | None = None


def _unquote_git_path(raw: str) -> str:
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        inner = raw[1:-1]
        return bytes(inner, "utf-8").decode("unicode_escape")
    return raw


def _clean_diff_path(raw: str, *, prefix_to_strip: str | None) -> str | None:
    text = raw.strip()
    if not text:
        return None
    if not text.startswith('"'):
        text = text.split("\t", 1)[0].strip()
    text = _unquote_git_path(text)
    if text in {"", "/dev/null", "a/dev/null", "b/dev/null", "dev/null"}:
        return None
    if prefix_to_strip and text.startswith(prefix_to_strip):
        text = text[len(prefix_to_strip) :]
    normalized = PurePosixPath(text).as_posix()
    if normalized in {"", "."}:
        return None
    return normalized


def _parse_git_diff_line(line: str) -> tuple[str | None, str | None]:
    rest = line[len("diff --git ") :].strip()
    if rest.startswith('"'):
        parts = re.findall(r'"(?:\\.|[^"\\])*"|\S+', rest)
        if len(parts) == 2:
            return (
                _clean_diff_path(parts[0], prefix_to_strip="a/"),
                _clean_diff_path(parts[1], prefix_to_strip="b/"),
            )
    matched = _GIT_DIFF_AB_RE.match(line)
    if matched:
        left, right = matched.group(1), matched.group(2)
        if left == right or (" b/" not in left):
            return (
                _clean_diff_path(left, prefix_to_strip=None),
                _clean_diff_path(right, prefix_to_strip=None),
            )
    return None, None


def extract_patch_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    old_rem = 0
    new_rem = 0

    def add(candidate: str | None) -> None:
        if candidate is not None and candidate not in seen:
            seen.add(candidate)
            paths.append(candidate)

    for line in patch.splitlines():
        if old_rem > 0 or new_rem > 0:
            if line.startswith("diff --git "):
                old_rem = 0
                new_rem = 0
            elif line.startswith("\\"):
                continue
            elif (line == "" or line.startswith(" ")) and old_rem > 0 and new_rem > 0:
                old_rem -= 1
                new_rem -= 1
                continue
            elif line.startswith("-") and old_rem > 0:
                old_rem -= 1
                continue
            elif line.startswith("+") and new_rem > 0:
                new_rem -= 1
                continue
            else:
                old_rem = 0
                new_rem = 0

        hunk = _HUNK_HEADER_RE.match(line)
        if hunk is not None:
            old_rem = int(hunk.group(2)) if hunk.group(2) is not None else 1
            new_rem = int(hunk.group(4)) if hunk.group(4) is not None else 1
            continue

        if line.startswith("diff --git "):
            left, right = _parse_git_diff_line(line)
            add(left)
            add(right)
        elif line.startswith("rename from "):
            add(_clean_diff_path(line[len("rename from ") :], prefix_to_strip=None))
        elif line.startswith("rename to "):
            add(_clean_diff_path(line[len("rename to ") :], prefix_to_strip=None))
        elif line.startswith("--- "):
            add(_clean_diff_path(line[4:], prefix_to_strip="a/"))
        elif line.startswith("+++ "):
            add(_clean_diff_path(line[4:], prefix_to_strip="b/"))

    return tuple(paths)


def gold_areas_for_patch(
    patch: str, repo_areas: Mapping[str, Iterable[str]]
) -> tuple[str, ...]:
    changed_paths = extract_patch_paths(patch)
    return tuple(sorted(areas_for_diff(changed_paths, repo_areas)))


def extract_issue_text(task: TaskItem) -> tuple[str, str]:
    if task.title and task.title.strip():
        return task.title.strip(), task.problem_statement.strip()
    cleaned = task.problem_statement.lstrip("\r\n")
    first_line, sep, remainder = cleaned.partition("\n")
    title = first_line.strip()
    body = remainder.lstrip("\r\n").rstrip() if sep else ""
    return title, body


def _row_to_task(row: Mapping[str, Any]) -> TaskItem:
    created = row.get("created_at")
    return TaskItem(
        instance_id=str(row["instance_id"]),
        repo=str(row["repo"]),
        problem_statement=str(row.get("problem_statement") or ""),
        patch=str(row.get("patch") or ""),
        base_commit=str(row["base_commit"]) if row.get("base_commit") else None,
        created_at=str(created) if created is not None else None,
        title=str(row["title"]) if row.get("title") else None,
    )


def load_parquet_bundle(
    *,
    parquet_dir: Path,
    repo: str,
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str = "local",
) -> DatasetBundle:
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found in {parquet_dir}")

    by_split: dict[str, dict[str, TaskItem]] = {}
    for file_path in parquet_files:
        split = file_path.name.split("-", 1)[0]
        table = pq.read_table(file_path, filters=[("repo", "=", repo)])
        bucket = by_split.setdefault(split, {})
        for row in table.to_pylist():
            item = _row_to_task(row)
            bucket[item.instance_id] = item

    split_counts = {split: len(items) for split, items in sorted(by_split.items())}
    primary_items: dict[str, TaskItem] = {}
    for split in ("test", *sorted(k for k in by_split if k != "test")):
        for instance_id, item in by_split.get(split, {}).items():
            primary_items.setdefault(instance_id, item)

    filtered_subset: bool | None = None
    if "test" in by_split and "filtered" in by_split:
        filtered_subset = set(by_split["filtered"]).issubset(set(by_split["test"]))

    ordered_tasks = tuple(primary_items[key] for key in sorted(primary_items))
    return DatasetBundle(
        dataset_id=dataset_id,
        revision=revision,
        repo=repo,
        tasks=ordered_tasks,
        split_counts=split_counts,
        filtered_subset_of_test=filtered_subset,
    )


def load_swe_rebench_tasks(
    *,
    repo: str = DEFAULT_REPO,
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str | None = None,
    cache_dir: Path,
    parquet_dir: Path | None = None,
    http: httpx.Client | None = None,
) -> DatasetBundle:
    extracted_root = cache_dir / "swe-rebench"
    extracted_root.mkdir(parents=True, exist_ok=True)
    safe_repo = repo.replace("/", "__")

    own_client = http is None
    client = http or httpx.Client(follow_redirects=True, timeout=120)
    try:
        resolved_rev = revision
        siblings: list[str] = []
        if resolved_rev is None or parquet_dir is None:
            info_url = f"https://huggingface.co/api/datasets/{dataset_id}"
            if resolved_rev:
                info_url = f"{info_url}/revision/{resolved_rev}"
            resp = client.get(info_url)
            resp.raise_for_status()
            meta = resp.json()
            resolved_rev = str(meta["sha"])
            siblings = [
                str(entry["rfilename"])
                for entry in meta.get("siblings", [])
                if str(entry.get("rfilename", "")).endswith(".parquet")
            ]

        assert resolved_rev is not None
        cached_json = extracted_root / resolved_rev / f"{safe_repo}.json"
        if cached_json.exists():
            raw = json.loads(cached_json.read_text(encoding="utf-8"))
            return DatasetBundle(
                dataset_id=str(raw["dataset_id"]),
                revision=str(raw["revision"]),
                repo=str(raw["repo"]),
                tasks=tuple(_row_to_task(item) for item in raw["tasks"]),
                split_counts={
                    str(k): int(v) for k, v in raw.get("split_counts", {}).items()
                },
                filtered_subset_of_test=raw.get("filtered_subset_of_test"),
            )

        target_parquet_dir = parquet_dir or (extracted_root / resolved_rev / "parquet")
        target_parquet_dir.mkdir(parents=True, exist_ok=True)
        for rel_path in siblings:
            local_file = target_parquet_dir / Path(rel_path).name
            if not local_file.exists():
                dl_url = (
                    f"https://huggingface.co/datasets/{dataset_id}/resolve/"
                    f"{resolved_rev}/{rel_path}"
                )
                tmp_file = local_file.with_suffix(".parquet.part")
                with client.stream("GET", dl_url) as stream:
                    stream.raise_for_status()
                    with tmp_file.open("wb") as handle:
                        for chunk in stream.iter_bytes(1024 * 1024):
                            handle.write(chunk)
                tmp_file.replace(local_file)

        bundle = load_parquet_bundle(
            parquet_dir=target_parquet_dir,
            repo=repo,
            dataset_id=dataset_id,
            revision=resolved_rev,
        )
        cached_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_id": bundle.dataset_id,
            "revision": bundle.revision,
            "repo": bundle.repo,
            "split_counts": bundle.split_counts,
            "filtered_subset_of_test": bundle.filtered_subset_of_test,
            "tasks": [
                {
                    "instance_id": t.instance_id,
                    "repo": t.repo,
                    "problem_statement": t.problem_statement,
                    "patch": t.patch,
                    "base_commit": t.base_commit,
                    "created_at": t.created_at,
                    "title": t.title,
                }
                for t in bundle.tasks
            ],
        }
        tmp_json = cached_json.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_json.replace(cached_json)
        return bundle
    finally:
        if own_client:
            client.close()


def score_task(
    task: TaskItem,
    outcome: ModelOutcome | None,
    *,
    repo_areas: Mapping[str, Iterable[str]],
    allowed_areas: tuple[str, ...],
) -> TaskOutcomeRecord:
    changed_paths = extract_patch_paths(task.patch)
    gold_areas = tuple(sorted(areas_for_diff(changed_paths, repo_areas)))

    if outcome is None or outcome.outcome != "completed":
        return TaskOutcomeRecord(
            instance_id=task.instance_id,
            changed_paths=changed_paths,
            gold_areas=gold_areas,
            call_outcome="not_run" if outcome is None else outcome.outcome,
            predicted_area=None,
            predicted_kind=None,
            predicted_priority=None,
            parse_failure=False,
            verdict="call_failed",
            model=None
            if outcome is None
            else (outcome.model_reported or outcome.model_pinned),
            tokens_in=0 if outcome is None else outcome.tokens_in,
            tokens_out=0 if outcome is None else outcome.tokens_out,
            retry_count=0 if outcome is None else outcome.retry_count,
            error_code=None if outcome is None else outcome.error_code,
            http_status=None if outcome is None else outcome.http_status,
        )

    parsed = parse_triage(outcome.text, allowed_areas=allowed_areas)
    parse_failed = parsed is None
    predicted_area = None if parsed is None else parsed.area

    if not gold_areas:
        verdict: Verdict = "no_gold_area"
    elif parse_failed:
        verdict = "parse_failure"
    elif predicted_area in gold_areas:
        verdict = "hit"
    else:
        verdict = "miss"

    return TaskOutcomeRecord(
        instance_id=task.instance_id,
        changed_paths=changed_paths,
        gold_areas=gold_areas,
        call_outcome=outcome.outcome,
        predicted_area=predicted_area,
        predicted_kind=None if parsed is None else parsed.kind,
        predicted_priority=None if parsed is None else parsed.priority,
        parse_failure=parse_failed,
        verdict=verdict,
        model=outcome.model_reported or outcome.model_pinned,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
        retry_count=outcome.retry_count,
        error_code=outcome.error_code,
        http_status=outcome.http_status,
    )


def _compute_marshmallow_layout_diagnostic(
    records: Sequence[TaskOutcomeRecord],
    repo_areas: Mapping[str, Iterable[str]],
) -> dict[str, Any] | None:
    legacy_records = [
        rec
        for rec in records
        if any(path.startswith("marshmallow/") for path in rec.changed_paths)
    ]
    if not legacy_records:
        return None

    norm_scorable = 0
    norm_hits = 0
    norm_no_gold = 0
    code_only_multi = 0
    src_mapped_docs_only = 0
    for rec in records:
        if any(
            path.startswith("marshmallow/") for path in rec.changed_paths
        ) and rec.gold_areas == ("docs-build",):
            src_mapped_docs_only += 1
        mapped_paths = [
            f"src/{path}" if path.startswith("marshmallow/") else path
            for path in rec.changed_paths
        ]
        mapped_gold = areas_for_diff(mapped_paths, repo_areas)
        code_gold = mapped_gold - {"docs-build"}
        if len(code_gold) > 1:
            code_only_multi += 1
        if rec.call_outcome != "completed":
            continue
        if not mapped_gold:
            norm_no_gold += 1
            continue
        norm_scorable += 1
        if rec.predicted_area is not None and rec.predicted_area in mapped_gold:
            norm_hits += 1

    return {
        "legacy_prefix_task_count": len(legacy_records),
        "legacy_docs_only_gold_count": src_mapped_docs_only,
        "mapped_code_multi_area_count": code_only_multi,
        "mapped_scorable": norm_scorable,
        "mapped_no_gold_area": norm_no_gold,
        "mapped_hits": norm_hits,
        "mapped_accuracy_scorable": (
            norm_hits / norm_scorable if norm_scorable > 0 else None
        ),
    }


def aggregate_eval(
    *,
    dataset_id: str,
    dataset_revision: str,
    repo: str,
    model_pinned: str,
    provider: str,
    evaluated_at: str,
    split_counts: dict[str, int],
    filtered_subset_of_test: bool | None,
    configured_areas: tuple[str, ...],
    tasks: Sequence[TaskItem],
    records: Sequence[TaskOutcomeRecord],
    repo_areas: Mapping[str, Iterable[str]] | None = None,
    blocker: str | None = None,
) -> EvalSummary:
    total_tasks = len(tasks)
    attempted = sum(1 for r in records if r.call_outcome == "completed")
    failed_calls = sum(1 for r in records if r.verdict == "call_failed")
    no_gold_area = sum(1 for r in records if r.verdict == "no_gold_area")
    hits = sum(1 for r in records if r.verdict == "hit")
    misses = sum(1 for r in records if r.verdict == "miss")
    parse_failures = sum(1 for r in records if r.parse_failure)
    scorable_parse_failures = sum(1 for r in records if r.verdict == "parse_failure")
    scorable = hits + misses + scorable_parse_failures
    multi_area_gold_count = sum(1 for r in records if len(r.gold_areas) > 1)

    accuracy_scorable = (hits / scorable) if scorable > 0 else None
    accuracy_all_attempted = (hits / attempted) if attempted > 0 else None

    all_area_names = (*configured_areas, "unknown")
    area_stats: list[AreaStats] = []
    completed_records = [r for r in records if r.call_outcome == "completed"]
    scorable_records = [r for r in completed_records if r.gold_areas]
    no_gold_records = [r for r in completed_records if not r.gold_areas]

    for area in all_area_names:
        in_gold = [r for r in scorable_records if area in r.gold_areas]
        pred_scorable = [r for r in scorable_records if r.predicted_area == area]
        pred_no_gold = [r for r in no_gold_records if r.predicted_area == area]
        direct_hits = sum(1 for r in pred_scorable if area in r.gold_areas)
        gold_task_hits = sum(1 for r in in_gold if r.verdict == "hit")
        gold_misses = sum(1 for r in in_gold if r.verdict != "hit")
        false_positives = sum(1 for r in pred_scorable if area not in r.gold_areas)
        area_stats.append(
            AreaStats(
                area=area,
                gold_support=len(in_gold),
                predicted_scorable=len(pred_scorable),
                predicted_no_gold=len(pred_no_gold),
                direct_hits=direct_hits,
                gold_task_hits=gold_task_hits,
                gold_misses=gold_misses,
                false_positives=false_positives,
            )
        )

    diagnostic = (
        _compute_marshmallow_layout_diagnostic(records, repo_areas)
        if repo_areas is not None and attempted > 0
        else None
    )

    return EvalSummary(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        repo=repo,
        model_pinned=model_pinned,
        provider=provider,
        evaluated_at=evaluated_at,
        split_counts=dict(split_counts),
        filtered_subset_of_test=filtered_subset_of_test,
        total_tasks=total_tasks,
        attempted=attempted,
        scorable=scorable,
        hits=hits,
        misses=misses,
        parse_failures=parse_failures,
        no_gold_area=no_gold_area,
        failed_calls=failed_calls,
        multi_area_gold_count=multi_area_gold_count,
        accuracy_scorable=accuracy_scorable,
        accuracy_all_attempted=accuracy_all_attempted,
        area_stats=tuple(area_stats),
        records=tuple(records),
        blocker=blocker,
        layout_diagnostic=diagnostic,
    )


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_gold(gold_areas: tuple[str, ...]) -> str:
    return ", ".join(f"`{a}`" for a in gold_areas) if gold_areas else "*(empty)*"


def _fmt_pred(record: TaskOutcomeRecord) -> str:
    if record.call_outcome != "completed":
        return "*(not completed)*"
    if record.parse_failure or record.predicted_area is None:
        return "*(parse_failure)*"
    return f"`{record.predicted_area}`"


def render_markdown_report(summary: EvalSummary) -> str:
    lines: list[str] = [
        "# Phase 3c — Offline Triage Accuracy Evaluation",
        "",
        "## Run Metadata",
        "",
        f"- **Dataset:** `{summary.dataset_id}`",
        f"- **Dataset revision:** `{summary.dataset_revision}`",
        f"- **Target repository:** `{summary.repo}`",
        f"- **Unique tasks:** `{summary.total_tasks}`"
        + (
            f" (`test`: {summary.split_counts.get('test', 0)}, "
            f"`filtered`: {summary.split_counts.get('filtered', 0)}; "
            f"`filtered` subset of `test`: `{summary.filtered_subset_of_test}`)"
            if summary.split_counts
            else ""
        ),
        f"- **Pinned model:** `{summary.model_pinned}` (provider: `{summary.provider}`)",
        f"- **Evaluated at (UTC):** `{summary.evaluated_at}`",
        "",
    ]

    if summary.blocker is not None:
        lines.extend(
            [
                "## Blocker",
                "",
                f"Live evaluation could not run: `{summary.blocker}`.",
                "No model metrics were fabricated.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Overall Accuracy & Scoring Rules",
            "",
            "| Metric | Count / Rate | Formula / Rule |",
            "|---|---:|---|",
            (
                f"| **Accuracy (scorable tasks)** | "
                f"**{summary.hits} / {summary.scorable} ({_fmt_pct(summary.accuracy_scorable)})** | "
                f"`hits / (attempted - no_gold_area)`; parse failures count as misses |"
            ),
            (
                f"| **Accuracy (all attempted tasks)** | "
                f"**{summary.hits} / {summary.attempted} ({_fmt_pct(summary.accuracy_all_attempted)})** | "
                f"`hits / attempted`; both parse failures and `no_gold_area` in denominator with 0 hits |"
            ),
            f"| Attempted / Total tasks | {summary.attempted} / {summary.total_tasks} | Completed triager model calls |",
            f"| Hits | {summary.hits} | `predicted_area in gold_areas` (one primary area suffices on multi-area diffs) |",
            f"| Misses (wrong area on non-empty gold) | {summary.misses} | `predicted_area not in gold_areas` |",
            f"| Parse failures | {summary.parse_failures} | `parse_triage(...) is None`; recorded as `None` (not fabricated) and counted as miss |",
            f"| No gold area (`empty gold` bucket) | {summary.no_gold_area} | `areas_for_diff` returned `set()`; never counted as a hit |",
            f"| Multi-area gold tasks (strict `agentteam.yaml`) | {summary.multi_area_gold_count} | `len(gold_areas) > 1` |",
            f"| Failed / skipped calls | {summary.failed_calls} | Non-completed model calls |",
            "",
            "### Treatment of Edge Cases",
            "",
            "1. **Gold area set:** Computed strictly via `areas_for_diff(extract_patch_paths(patch), marshmallow.areas)` using `agentteam.yaml`.",
            "2. **Multi-area patches:** A task is a **hit** when `TriageResult.area` matches any member of its non-empty gold set.",
            "3. **Parse failures:** When `parse_triage` returns `None`, `predicted_area` is recorded as `None` (`parse_failure`) rather than fabricating an area, and counted as a miss on scorable tasks.",
            "4. **Empty gold set (`no_gold_area`):** Tasks where no `agentteam.yaml` path glob matched the gold patch are placed in the separate `no_gold_area` bucket and never counted as hits (even if the model predicts `unknown`, since `unknown` is never in `areas_for_diff`). Both `hits / scorable` and `hits / attempted` are reported above.",
            "",
        ]
    )

    if summary.layout_diagnostic is not None:
        diag = summary.layout_diagnostic
        lines.extend(
            [
                "### Path-Layout Observation (`marshmallow/` vs `src/marshmallow/`)",
                "",
                (
                    f"`agentteam.yaml` configures current `src/marshmallow/*.py` globs plus `docs-build` (`docs/**`, `CHANGELOG.rst`, `pyproject.toml`, `.github/workflows/**`). "
                    f"In `{summary.dataset_id}`, **{diag['legacy_prefix_task_count']}** older marshmallow tasks predate the repository's move from `marshmallow/` to `src/marshmallow/`: "
                    f"those without `CHANGELOG.rst`/`docs/**` edits land in `no_gold_area`, while **{diag['legacy_docs_only_gold_count']}** legacy code fixes that also updated `CHANGELOG.rst` or `docs/**` receive `gold_areas = ['docs-build']` only under raw `agentteam.yaml` globs. "
                    f"As a diagnostic reference (without altering primary scoring or `agentteam.yaml`), mapping legacy `marshmallow/<path>` prefixes to `src/marshmallow/<path>` yields "
                    f"**{diag['mapped_hits']} / {diag['mapped_scorable']} ({_fmt_pct(diag['mapped_accuracy_scorable'])})** on scorable tasks "
                    f"({diag['mapped_no_gold_area']} empty-gold tasks; {diag['mapped_code_multi_area_count']} multi-code-area tasks)."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Per-Area Breakdown (Scorable Tasks)",
            "",
            "| Area | Gold Support | Predicted (Scorable) | Direct Hits | Gold Task Hits (incl. multi-area) | Gold Misses | Wrong Predictions (FP) | Predicted on `no_gold_area` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stat in summary.area_stats:
        lines.append(
            f"| `{stat.area}` | {stat.gold_support} | {stat.predicted_scorable} | "
            f"{stat.direct_hits} | {stat.gold_task_hits} | {stat.gold_misses} | "
            f"{stat.false_positives} | {stat.predicted_no_gold} |"
        )
    lines.append("")

    non_hits = [r for r in summary.records if r.verdict != "hit"]
    lines.extend(
        [
            f"## Confusion & Non-Hit Breakdown ({len(non_hits)} tasks)",
            "",
            "| Instance ID | Gold Area Set | Predicted Area | Verdict | Changed Paths |",
            "|---|---|---|---|---|",
        ]
    )
    if non_hits:
        for rec in non_hits:
            paths_str = ", ".join(f"`{p}`" for p in rec.changed_paths) or "*(none)*"
            lines.append(
                f"| `{rec.instance_id}` | {_fmt_gold(rec.gold_areas)} | "
                f"{_fmt_pred(rec)} | `{rec.verdict}` | {paths_str} |"
            )
    else:
        lines.append("| *(none)* | — | — | — | — |")
    lines.append("")

    failed = [r for r in summary.records if r.verdict == "call_failed"]
    lines.extend(
        [
            "## Skipped / Failed Calls",
            "",
        ]
    )
    if failed:
        lines.extend(
            [
                "| Instance ID | Call Outcome | Error Code | HTTP Status |",
                "|---|---|---|---:|",
            ]
        )
        for rec in failed:
            lines.append(
                f"| `{rec.instance_id}` | `{rec.call_outcome}` | "
                f"`{rec.error_code or 'none'}` | {rec.http_status or '—'} |"
            )
    else:
        lines.append("None — all 61 tasks completed on the pinned model.")
    lines.append("")

    lines.extend(
        [
            f"## Complete Task Results ({len(summary.records)} tasks)",
            "",
            "| Instance ID | Gold Area Set | Predicted Area | Verdict |",
            "|---|---|---|---|",
        ]
    )
    for rec in summary.records:
        verdict_label = "hit" if rec.verdict == "hit" else f"**{rec.verdict}**"
        lines.append(
            f"| `{rec.instance_id}` | {_fmt_gold(rec.gold_areas)} | "
            f"{_fmt_pred(rec)} | {verdict_label} |"
        )
    lines.append("")
    return "\n".join(lines)


def summary_to_json_dict(summary: EvalSummary) -> dict[str, Any]:
    return {
        "dataset_id": summary.dataset_id,
        "dataset_revision": summary.dataset_revision,
        "repo": summary.repo,
        "model_pinned": summary.model_pinned,
        "provider": summary.provider,
        "evaluated_at": summary.evaluated_at,
        "split_counts": summary.split_counts,
        "filtered_subset_of_test": summary.filtered_subset_of_test,
        "blocker": summary.blocker,
        "total_tasks": summary.total_tasks,
        "attempted": summary.attempted,
        "scorable": summary.scorable,
        "hits": summary.hits,
        "misses": summary.misses,
        "parse_failures": summary.parse_failures,
        "no_gold_area": summary.no_gold_area,
        "failed_calls": summary.failed_calls,
        "multi_area_gold_count": summary.multi_area_gold_count,
        "accuracy_scorable": summary.accuracy_scorable,
        "accuracy_all_attempted": summary.accuracy_all_attempted,
        "layout_diagnostic": summary.layout_diagnostic,
        "area_stats": [
            {
                "area": s.area,
                "gold_support": s.gold_support,
                "predicted_scorable": s.predicted_scorable,
                "predicted_no_gold": s.predicted_no_gold,
                "direct_hits": s.direct_hits,
                "gold_task_hits": s.gold_task_hits,
                "gold_misses": s.gold_misses,
                "false_positives": s.false_positives,
            }
            for s in summary.area_stats
        ],
        "records": [
            {
                "instance_id": r.instance_id,
                "changed_paths": list(r.changed_paths),
                "gold_areas": list(r.gold_areas),
                "call_outcome": r.call_outcome,
                "predicted_area": r.predicted_area,
                "predicted_kind": r.predicted_kind,
                "predicted_priority": r.predicted_priority,
                "parse_failure": r.parse_failure,
                "verdict": r.verdict,
                "model": r.model,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "retry_count": r.retry_count,
                "error_code": r.error_code,
                "http_status": r.http_status,
            }
            for r in summary.records
        ],
    }


def _cache_entry_key(
    *,
    dataset_id: str,
    dataset_revision: str,
    repo: str,
    model_pinned: str,
    instance_id: str,
    messages: list[dict[str, str]],
) -> str:
    msg_digest = sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{dataset_id}|{dataset_revision}|{repo}|{model_pinned}|{instance_id}|{msg_digest}"


def _load_outcome_cache(path: Path) -> dict[str, ModelOutcome]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("outcomes", {}) if isinstance(raw, dict) else {}
    loaded: dict[str, ModelOutcome] = {}
    for key, payload in entries.items():
        outcome = ModelOutcome.model_validate(payload)
        if outcome.outcome == "completed":
            loaded[str(key)] = outcome
    return loaded


def _save_outcome_cache(path: Path, cache: Mapping[str, ModelOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "outcomes": {
            key: outcome.model_dump(mode="json")
            for key, outcome in sorted(cache.items())
            if outcome.outcome == "completed"
        }
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_triage_eval(
    *,
    tasks: Sequence[TaskItem],
    dataset_id: str,
    dataset_revision: str,
    repo: str,
    settings: Settings,
    cache_path: Path,
    split_counts: dict[str, int] | None = None,
    filtered_subset_of_test: bool | None = None,
    config_path: str | Path = "agentteam.yaml",
    max_wait_s: float = 180.0,
    max_task_attempts: int = 8,
    transport: httpx.BaseTransport | None = None,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int, TaskOutcomeRecord, bool], None] | None = None,
) -> EvalSummary:
    clock = now or (lambda: datetime.now(UTC))
    repo_cfg = load_agentteam_config(config_path).repo(repo)
    pin = pin_for("triager", settings)
    evaluated_at = clock().astimezone(UTC).isoformat().replace("+00:00", "Z")

    if not settings.groq_api_key.strip():
        empty_records = tuple(
            score_task(
                task,
                None,
                repo_areas=repo_cfg.areas,
                allowed_areas=repo_cfg.allowed_area_names,
            )
            for task in tasks
        )
        return aggregate_eval(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            repo=repo,
            model_pinned=pin.model,
            provider=pin.pool,
            evaluated_at=evaluated_at,
            split_counts=split_counts or {},
            filtered_subset_of_test=filtered_subset_of_test,
            configured_areas=tuple(repo_cfg.areas),
            tasks=tasks,
            records=empty_records,
            repo_areas=repo_cfg.areas,
            blocker="MissingGroqApiKey",
        )

    outcome_cache = _load_outcome_cache(cache_path)
    store = Store(settings)
    ledger = Ledger(settings, store)
    http_client = (
        httpx.Client(transport=transport) if transport is not None else httpx.Client()
    )
    wait_cap = min(max_wait_s, settings.triager_wait_cap_s)
    records: list[TaskOutcomeRecord] = []
    abort_remaining = False

    try:
        model_client = ModelClient(
            settings=settings,
            ledger=ledger,
            store=store,
            http=http_client,
            now=clock,
            monotonic=monotonic,
            sleep=sleep,
        )
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            title, body = extract_issue_text(task)
            messages = triage_messages(
                title, body, allowed_areas=repo_cfg.allowed_area_names
            )
            cache_key = _cache_entry_key(
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                repo=repo,
                model_pinned=pin.model,
                instance_id=task.instance_id,
                messages=messages,
            )
            cached = outcome_cache.get(cache_key)
            if cached is not None:
                rec = score_task(
                    task,
                    cached,
                    repo_areas=repo_cfg.areas,
                    allowed_areas=repo_cfg.allowed_area_names,
                )
                records.append(rec)
                if progress is not None:
                    progress(index, total, rec, True)
                continue

            if abort_remaining:
                rec = score_task(
                    task,
                    None,
                    repo_areas=repo_cfg.areas,
                    allowed_areas=repo_cfg.allowed_area_names,
                )
                records.append(rec)
                if progress is not None:
                    progress(index, total, rec, False)
                continue

            last_outcome: ModelOutcome | None = None
            request = ModelCallRequest(
                run_id=f"eval:triage:{task.instance_id}",
                step="triage",
                role="triager",
                messages=messages,
                sampling=Sampling(),
                taint=2,
                source="github_issue",
            )
            for attempt_idx in range(1, max_task_attempts + 1):
                outcome = model_client.call(request)
                last_outcome = outcome

                if outcome.outcome == "completed":
                    outcome_cache[cache_key] = outcome
                    _save_outcome_cache(cache_path, outcome_cache)
                    break

                if outcome.outcome == "throttled":
                    wait_s = outcome.retry_in if outcome.retry_in is not None else 5.0
                    if wait_s > wait_cap or attempt_idx == max_task_attempts:
                        if wait_s > wait_cap:
                            abort_remaining = True
                        break
                    sleep(wait_s + 0.5)
                    continue

                if outcome.outcome == "quota_exhausted":
                    if outcome.retry_at is None:
                        abort_remaining = True
                        break
                    wait_s = max(0.0, (outcome.retry_at - clock()).total_seconds())
                    if wait_s > wait_cap or attempt_idx == max_task_attempts:
                        abort_remaining = True
                        break
                    sleep(wait_s + 0.5)
                    continue

                # Non-retryable failure (e.g., 400/401/403/model_mismatch)
                if outcome.http_status in (401, 403):
                    abort_remaining = True
                break

            rec = score_task(
                task,
                last_outcome,
                repo_areas=repo_cfg.areas,
                allowed_areas=repo_cfg.allowed_area_names,
            )
            records.append(rec)
            if progress is not None:
                progress(index, total, rec, False)
    finally:
        http_client.close()

    blocker_class: str | None = None
    if all(r.verdict == "call_failed" for r in records) and records:
        first_err = records[0].error_code or records[0].call_outcome
        blocker_class = f"LiveModelCallBlocked:{first_err}"

    return aggregate_eval(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        repo=repo,
        model_pinned=pin.model,
        provider=pin.pool,
        evaluated_at=evaluated_at,
        split_counts=split_counts or {},
        filtered_subset_of_test=filtered_subset_of_test,
        configured_areas=tuple(repo_cfg.areas),
        tasks=tasks,
        records=records,
        repo_areas=repo_cfg.areas,
        blocker=blocker_class,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline triage-accuracy evaluation on SWE-rebench tasks."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("eval/.cache"))
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("eval/results/triage-accuracy.md"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("eval/results/triage-accuracy.json"),
    )
    parser.add_argument("--dotenv-path", type=Path, default=Path(".env"))
    parser.add_argument("--max-wait-s", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.dotenv_path.exists():
        load_dotenv(dotenv_path=args.dotenv_path, override=False)

    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_settings = Settings.from_env()
    from dataclasses import replace

    settings = replace(
        base_settings,
        models_sqlite_path=str(cache_dir / "models.sqlite"),
    )

    bundle = load_swe_rebench_tasks(
        repo=args.repo,
        dataset_id=args.dataset,
        revision=args.revision,
        cache_dir=cache_dir,
        parquet_dir=args.parquet_dir,
    )

    def log_progress(
        idx: int, total: int, rec: TaskOutcomeRecord, from_cache: bool
    ) -> None:
        source_tag = "cache" if from_cache else "live"
        print(
            f"[{idx:02d}/{total:02d}] ({source_tag}) {rec.instance_id}: "
            f"gold={list(rec.gold_areas)} pred={rec.predicted_area} -> {rec.verdict}",
            file=sys.stderr,
            flush=True,
        )

    summary = run_triage_eval(
        tasks=bundle.tasks,
        dataset_id=bundle.dataset_id,
        dataset_revision=bundle.revision,
        repo=bundle.repo,
        settings=settings,
        cache_path=cache_dir / "triage-outcomes.json",
        split_counts=bundle.split_counts,
        filtered_subset_of_test=bundle.filtered_subset_of_test,
        max_wait_s=args.max_wait_s,
        progress=log_progress,
    )

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_markdown_report(summary), encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(summary_to_json_dict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    if summary.blocker is not None:
        print(
            f"Blocked ({summary.blocker}); wrote blocker report to {args.report_md}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Completed {summary.attempted}/{summary.total_tasks} tasks: "
        f"scorable accuracy {summary.hits}/{summary.scorable} "
        f"({_fmt_pct(summary.accuracy_scorable)}), "
        f"all-attempted {summary.hits}/{summary.attempted} "
        f"({_fmt_pct(summary.accuracy_all_attempted)}). "
        f"Report: {args.report_md}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
