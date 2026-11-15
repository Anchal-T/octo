"""P1 thresholds and a report that never turns missing live data into a pass."""

import argparse
import json
from pathlib import Path


def thresholds(conditions, invalidation, revert, lifecycle):
    c1, c2 = conditions["C1"]["stale_usage"], conditions["C2"]["stale_usage"]
    cut = (c1 - c2) / c1 if c1 and c2 is not None else None
    values = [
        c2,
        cut,
        invalidation["precision"],
        invalidation["recall"],
        revert["passed"] / revert["total"] if revert["total"] else None,
        None,
        None,
        lifecycle["passed"] / lifecycle["total"] if lifecycle["total"] else None,
    ]
    checks = [
        lambda x: x <= 0.05,
        lambda x: x >= 0.5,
        lambda x: x >= 0.8,
        lambda x: x >= 0.7,
        lambda x: x == 1,
        lambda x: False,
        lambda x: False,
        lambda x: x == 1,
    ]
    labels = [
        "C2 stale usage <= 5%",
        "relative cut >= 50%",
        "precision >= .8",
        "recall >= .7",
        "revert recovery = 100%",
        "live task success: no drop",
        "live tokens <= +20%",
        "user_report lifecycle / TTL = 100%",
    ]
    return {
        f"T{i + 1}": {
            "status": "PASS" if v is not None and checks[i](v) else "FAIL",
            "value": v,
            "measured": v is not None,
            "criterion": labels[i],
            "note": "Live slice not run; no conclusion." if i in (5, 6) else "",
        }
        for i, v in enumerate(values)
    }


def markdown(result):
    window = result.get("window", {})
    lines = [
        "# P1 stale-fact retrieval replay",
        "",
        f"Repo: Anchal-T/sqlparse. Window: {len(window.get('main', []))} first-parent + {len(window.get('side', []))} natural side commits.",
        f"Harness commit: `{result.get('harness_commit', 'test fixture')}`. Seed: {result.get('seed', 1)}. Model requests: **0**.",
        "",
        "| Condition | Queries | Injected | Stale | Stale usage | Points with stale | Expired withheld |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["conditions"].items():
        rate = (
            "n/a" if row["stale_usage"] is None else f"{100 * row['stale_usage']:.2f}%"
        )
        lines.append(
            f"| {name} | {row['query_points']} | {row['injected']} | {row['stale_injected']} | {rate} | {row['points_with_stale']} | {row['expired_withheld']} |"
        )
    lines += [
        "",
        "| Type | Logical seeds | Closures | Truly stale transitions | Precision | Recall | Delay median / p90 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in result["by_type"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['closed']} | {row['truly_stale']} | {row['precision']} | {row['recall']} | {row['delay_median']} / {row['delay_p90']} |"
        )
    lines += ["", "| Threshold | Result | Value | Requirement |", "|---|---|---:|---|"]
    for name, row in result["thresholds"].items():
        lines.append(
            f"| {name} | {row['status']} | {row['value'] if row['measured'] else 'NOT RUN'} | {row['criterion']} {row['note']} |"
        )
    lines += [
        "",
        "T6/T7 are marked FAIL only because the requested live slice was not run; they are unmeasured, not evidence of regression.",
        "",
        f"Revert checks: {result['revert_checks']}. User-report checks: {result['lifecycle_checks']}.",
        f"Expired-but-true withdrawals: {result['expired_true']}. TTL is not counted as a closure.",
        "",
        "## Coverage and limitations",
        "",
    ]
    lines += ["- " + note for note in result.get("notes", [])]
    lines += [
        "",
        "## Proposed live slice (not run)",
        "",
        "Ten existing fork issue copies #5–#14, C0/C2 interleaved per issue, 20 runs on pinned qwen/qwen3.8-27b. The first two use 60cdc649726bf1bc4f1b336050560b336da715ec; the others use 09dbe1eec31ed1b3cd05366748865f11ba427d32. No upstream writes or merges.",
        "The successful #10 smoke run used 19 completion HTTP requests (including four invalid-tool calls and one 429), versus 14 successful completions. Estimated central budget: **380 Groq completion requests**, plus one model-list GET; planning range **280–600**. At the observed 28,049 input + 1,075 output tokens/run, 20 similar runs would use about **582,480 tokens**, at least **73 minutes** at 8,000 TPM and about **3 daily 200k-token budgets**. Other issues may need more turns or fail early. These are estimates from one issue, not measured slice costs.",
        "Before execution: add an explicit per-run C0/C2 selector (C0 skips retrieval), freeze a fact-pool snapshot so one condition cannot feed facts to the other, keep model retries counted, record resolve rate/tokens/latency/stale citations, and require approval to start. The live driver is not executed by this retrieval CLI.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    args.result.with_suffix(".md").write_text(
        markdown(json.loads(args.result.read_text()))
    )


if __name__ == "__main__":
    main()
