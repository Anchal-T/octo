from datetime import datetime

from .dag import Dag
from .store import MemoryStore


class TimeLinearProjector:
    def __init__(self, dag: Dag, store: MemoryStore) -> None:
        self.dag = dag
        self.store = store

    def end_for(
        self, fact_id: str, closing_commit: str, main_head: str
    ) -> datetime | None:
        if closing_commit not in self.dag.commits:
            raise KeyError(closing_commit)
        for sha in self.dag.first_parent_path(main_head):
            if closing_commit in self.dag.ancestors(sha):
                committed_at = self.dag.commits[sha].committed_at
                self.store.end_fact_valid_time([fact_id], committed_at)
                return committed_at
        return None
