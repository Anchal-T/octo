"""Rebuildable eval-only git branches and explicit seed ground truth."""

import ast
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

TYPES = ("rename", "move", "behavior", "revert", "nonancestor", "userreport")


def git(repo, *args, input=None, env=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input,
        text=True,
        capture_output=True,
        check=True,
        env=env,
    ).stdout.strip()


def window(repo, count=400, side_count=30):
    head = git(repo, "rev-parse", "HEAD")
    path = git(repo, "rev-list", "--first-parent", head).splitlines()
    main = path[:count][::-1]
    if len(main) != count:
        raise ValueError("not enough first-parent commits")
    topo = git(repo, "rev-list", "--topo-order", "--reverse", head).splitlines()
    parents = git(repo, "show", "-s", "--format=%P", main[0]).split()
    old = set(git(repo, "rev-list", parents[0]).splitlines()) if parents else set()
    candidates = [sha for sha in topo if sha not in set(path) and sha not in old]
    side = candidates[-side_count:] if side_count else []
    return {"head": head, "main": main, "side": side}


def commit(repo, parent, files, branch, ordinal):
    """Write objects/refs only; leave the checkout untouched and use fixed dates."""
    index = Path(repo).resolve().parent / "seed-index"
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    git(repo, "read-tree", parent, env=env)
    for path, source in sorted(files.items()):
        if source is None:
            git(repo, "update-index", "--force-remove", path, env=env)
        else:
            blob = git(repo, "hash-object", "-w", "--stdin", input=source)
            git(
                repo,
                "update-index",
                "--add",
                "--cacheinfo",
                f"100644,{blob},{path}",
                env=env,
            )
    tree = git(repo, "write-tree", env=env)
    date = datetime.fromisoformat(
        git(repo, "show", "-s", "--format=%cI", parent)
    ) + timedelta(seconds=1)
    author = {
        **env,
        "GIT_AUTHOR_NAME": "P1 eval fixture",
        "GIT_AUTHOR_EMAIL": "p1-eval@example.invalid",
        "GIT_COMMITTER_NAME": "P1 eval fixture",
        "GIT_COMMITTER_EMAIL": "p1-eval@example.invalid",
        "GIT_AUTHOR_DATE": date.isoformat(),
        "GIT_COMMITTER_DATE": date.isoformat(),
    }
    sha = git(
        repo,
        "commit-tree",
        tree,
        "-p",
        parent,
        input=f"Eval-only {branch} stage {ordinal}\n",
        env=author,
    )
    ref = "refs/heads/" + branch
    try:
        existing = git(repo, "rev-parse", "--verify", ref)
    except subprocess.CalledProcessError:
        existing = None
    if existing and existing != sha:
        common = git(repo, "merge-base", existing, sha)
        if common not in (existing, sha):
            raise ValueError(f"refusing to replace unrelated local branch {branch}")
        if common == sha:
            index.unlink(missing_ok=True)
            return sha
    git(repo, "update-ref", ref, sha)
    index.unlink(missing_ok=True)
    return sha


def source(ty, n, changed=False):
    lines = [
        "# Synthetic P1 eval fixture: not a sqlparse product feature.",
        "class Seeds:",
    ]
    for i in range(n):
        name = f"f{i:02}" if not (ty == "rename" and changed) else f"renamed{i:02}"
        value = i + (1000 if changed and ty in ("behavior", "revert") else 0)
        lines.extend([f"    def {name}(self):", f"        return {value}"])
    if ty != "move":
        lines.extend(["    def control(self):", "        return -1"])
    return "\n".join(lines) + "\n"


def check_seed(repo, sha, path, name, value, *, static=False):
    """Executable seed evidence: return true iff the recorded claim still holds."""
    try:
        content = git(repo, "show", sha + ":" + path)
    except subprocess.CalledProcessError:
        return False
    if static:
        return name in definitions(content)
    namespace = {}
    exec(compile(content, path, "exec"), namespace)  # noqa: S102 - generated fixture sources only
    method = getattr(namespace["Seeds"](), name, None)
    return method is not None and method() == value


def definitions(content):
    """Qualified names and structural bodies, independent of SCIP's differ."""
    found = {}

    def walk(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                # Ignore only the declaration's name when looking for a rename.
                # Ambiguous identical bodies are excluded by discover_natural.
                body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                found[name] = body
                walk(node.body, name + ".")

    walk(ast.parse(content).body)
    return found


def discover_natural(repo, commits):
    candidates = []
    for change in commits:
        parent = git(repo, "show", "-s", "--format=%P", change).split()[0]
        paths = git(
            repo, "diff", "--name-only", parent, change, "--", "sqlparse"
        ).splitlines()
        before, after = {}, {}
        for path in paths:
            if not path.endswith(".py"):
                continue
            for sha, target in [(parent, before), (change, after)]:
                try:
                    content = git(repo, "show", sha + ":" + path)
                    defs = definitions(content)
                except (subprocess.CalledProcessError, SyntaxError):
                    continue
                target.update({(path, name): body for name, body in defs.items()})
        removed = set(before) - set(after)
        added = set(after) - set(before)
        for old in sorted(removed):
            matches = [new for new in added if after[new] == before[old]]
            if len(matches) != 1 or sum(before[x] == before[old] for x in removed) != 1:
                continue
            new = matches[0]
            ty = "move" if old[0] != new[0] else "rename"
            module = old[0][:-3].replace("/", ".")
            anchor = module + "." + old[1]
            candidates.append(
                {
                    "type": ty,
                    "origin": parent,
                    "change": change,
                    "restore": None,
                    "anchor_names": [anchor],
                    "claim": f"Eval seed: {anchor} is defined in {old[0]}.",
                    "evidence": {
                        "path": old[0],
                        "name": old[1],
                        "value": None,
                        "static": True,
                    },
                    "construction": "natural",
                    "replacement": {"path": new[0], "name": new[1]},
                }
            )
    return candidates


def construct(repo, win, seed=1):
    rows = []
    branches = []
    synthetic = []
    natural_counts = {"rename": 0, "move": 0}
    for candidate in discover_natural(repo, win["main"]):
        ty = candidate["type"]
        if natural_counts[ty] == 20:
            continue
        candidate["id"] = f"{ty}-natural-{natural_counts[ty]:02}"
        natural_counts[ty] += 1
        rows.append(candidate)
    for pos, ty in enumerate(TYPES[:-1]):
        base = win["main"][min(len(win["main"]) - 1, (pos + 1) * len(win["main"]) // 6)]
        branch = f"eval/p1-{ty}-{seed:02}"
        n = 20 - natural_counts.get(ty, 0)
        path = f"sqlparse/_p1_eval_{ty}.py"
        files = {path: source(ty, n)}
        control_path = f"sqlparse/_p1_eval_{ty}_control.py"
        if ty == "move":
            files[control_path] = (
                "class Seeds:\n    def control(self):\n        return -1\n"
            )
        origin = commit(repo, base, files, branch, 1)
        synthetic.append(origin)
        change = restore = None
        if ty != "nonancestor":
            changes = {path: source(ty, n, True)}
            if ty == "move":
                changes = {path: None, path.replace(".py", "_moved.py"): source(ty, n)}
            change = commit(repo, origin, changes, branch, 2)
            synthetic.append(change)
        if ty == "revert":
            restore = commit(repo, change, {path: source(ty, n)}, branch, 3)
            synthetic.append(restore)
        branches.append(
            {"name": branch, "base": base, "tip": restore or change or origin}
        )
        for i in range(n):
            sid = f"{ty}-{i:02}"
            anchor = f"sqlparse._p1_eval_{ty}.Seeds.f{i:02}"
            rows.append(
                {
                    "id": sid,
                    "type": ty,
                    "origin": origin,
                    "change": change,
                    "restore": restore,
                    "anchor_names": [anchor],
                    "claim": f"Synthetic eval seed: {anchor} returns {i}.",
                    "evidence": {
                        "path": path,
                        "name": f"f{i:02}",
                        "value": i,
                        "static": False,
                    },
                    "construction": "synthetic",
                }
            )
        cpath = control_path if ty == "move" else path
        rows.append(
            {
                "id": ty + "-control",
                "type": "control",
                "origin": origin,
                "change": None,
                "restore": None,
                "anchor_names": [
                    f"sqlparse._p1_eval_{ty}"
                    + ("_control" if ty == "move" else "")
                    + ".Seeds.control"
                ],
                "claim": "Synthetic negative control: control returns -1.",
                "evidence": {
                    "path": cpath,
                    "name": "control",
                    "value": -1,
                    "static": False,
                },
                "construction": "synthetic",
            }
        )
    # Linked-fix reports start on real main; their unmerged fix demonstrates that
    # the same issue report remains visible on a base without that fix.
    base = win["main"][-4]
    branch = f"eval/p1-userreport-{seed:02}"
    closing = commit(
        repo,
        base,
        {"P1_EVAL_ONLY.txt": "Synthetic linked-fix lifecycle probe.\n"},
        branch,
        1,
    )
    synthetic.append(closing)
    branches.append({"name": branch, "base": base, "tip": closing})
    sibling_branch = f"eval/p1-userreport-{seed + 1:02}"
    sibling = commit(
        repo,
        base,
        {"P1_EVAL_ONLY_OTHER.txt": "Synthetic branch without the fix.\n"},
        sibling_branch,
        1,
    )
    synthetic.append(sibling)
    branches.append({"name": sibling_branch, "base": base, "tip": sibling})
    for i in range(10):
        rows.append(
            {
                "id": f"userreport-fix-{i:02}",
                "type": "userreport",
                "origin": base,
                "change": closing,
                "restore": None,
                "anchor_names": [],
                "claim": f"Synthetic user report {i}: example issue remains unresolved.",
                "source_issue": f"eval/sqlparse#{1000 + i}",
                "construction": "synthetic",
                "checks": [
                    {"base": base, "keep": True},
                    {"base": closing, "keep": False},
                    {"base": sibling, "keep": True},
                ],
            }
        )
    for i in range(10):
        offset = i * max(1, (len(win["main"]) - 51) // 10)
        origin = win["main"][offset]
        rows.append(
            {
                "id": f"userreport-ttl-{i:02}",
                "type": "userreport",
                "origin": origin,
                "change": None,
                "restore": None,
                "anchor_names": [],
                "claim": f"Synthetic aging report {i}: still true until the hard TTL.",
                "source_issue": f"eval/sqlparse#{2000 + i}",
                "construction": "synthetic",
                "checks": [
                    {
                        "base": win["main"][offset + j],
                        "keep": j == 49,
                        "isolate_ttl": True,
                    }
                    for j in (49, 50)
                ],
            }
        )
    for row in rows:
        if "evidence" in row:
            kwargs = row["evidence"]
            assert check_seed(repo, row["origin"], **kwargs), row
            if row["change"]:
                assert not check_seed(repo, row["change"], **kwargs), row
            if row["restore"]:
                assert check_seed(repo, row["restore"], **kwargs), row
    return {
        "seed": seed,
        "window": win,
        "branches": branches,
        "synthetic_commits": synthetic,
        "seeds": rows,
        "natural_seed_counts": natural_counts,
    }
