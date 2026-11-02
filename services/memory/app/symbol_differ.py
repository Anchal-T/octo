from collections.abc import Iterable

from .git_repo import Hunk
from .scip import ScipIndex


class SymbolDiffer:
    @staticmethod
    def diff(parent: ScipIndex, child: ScipIndex, hunks: Iterable[Hunk]) -> set[str]:
        changed = set(parent.symbols ^ child.symbols)
        for hunk in hunks:
            if hunk.old_len:
                start = hunk.old_start - 1
                changed.update(
                    parent.overlapping(hunk.file, start, start + hunk.old_len)
                )
            if hunk.new_len:
                start = hunk.new_start - 1
                changed.update(
                    child.overlapping(hunk.file, start, start + hunk.new_len)
                )
        return changed
