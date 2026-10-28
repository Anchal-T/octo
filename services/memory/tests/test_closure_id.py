import hashlib

from agentteam_common.ids import closure_id


def test_closure_id_is_stable_for_a_fact_and_closing_commit():
    digest = hashlib.sha256(b"fact-123|commit-456").hexdigest()[:20]

    assert closure_id("fact-123", "commit-456") == f"fc-{digest}"
