from collections import defaultdict
from pathlib import Path


def test_service_test_file_basenames_are_unique():
    services = Path(__file__).parents[2]
    paths = services.rglob("test_*.py")
    by_name = defaultdict(list)
    for path in paths:
        by_name[path.name].append(path)

    duplicates = {
        name: sorted(str(path) for path in matches)
        for name, matches in by_name.items()
        if len(matches) > 1
    }
    assert not duplicates, f"test file basename collisions: {duplicates}"
