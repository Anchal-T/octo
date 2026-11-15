"""Seed ground truth, deliberately independent of production validity code."""

from datetime import datetime


class Oracle:
    def __init__(self, dag, seeds):
        self.commits = dag.commits
        self.seeds = {fid: seed for seed in seeds for fid in seed["fact_ids"]}
        self._ancestors = {}

    def ancestors(self, sha):
        if sha not in self._ancestors:
            found, pending = set(), [sha]
            while pending:
                node = pending.pop()
                if node not in found:
                    found.add(node)
                    pending.extend(self.commits[node].parents)
            self._ancestors[sha] = found
        return self._ancestors[sha]

    def main_distance(self, origin, base):
        count, sha = 0, base
        while sha != origin:
            node = self.commits[sha]
            if not node.parents:
                return None
            count += node.branch == "main"
            sha = node.parents[0]
        return count

    def truth(self, fact, base):
        seed = self.seeds[fact["_id"]]
        if fact.get("kind") == "user_report":
            distance = self.main_distance(fact["valid_from_commit"], base)
            checked = fact.get("last_confirmed_at")
            if isinstance(checked, str):
                checked = datetime.fromisoformat(checked)
            age = max(
                0, (self.commits[base].committed_at - checked).total_seconds() / 86400
            )
            if (distance is not None and distance >= 50) or 0.8 * 2 ** (
                -age / 30
            ) < 0.6:
                return "expired"
        ancestors = self.ancestors(base)
        if fact["valid_from_commit"] not in ancestors:
            return "stale"
        if "truth_by_base" in seed:
            return "valid" if seed["truth_by_base"][base] else "stale"
        if seed.get("change") in ancestors:
            if seed.get("restore") in ancestors:
                return "valid"
            return "stale"
        return "valid"
