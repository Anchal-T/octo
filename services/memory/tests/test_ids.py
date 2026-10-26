from agentteam_common import stable_id


def test_stable_id_is_repeatable_and_scoped_by_prefix():
    assert stable_id("ev", "run-1:issue") == stable_id("ev", "run-1:issue")
    assert stable_id("ev", "run-1:issue") != stable_id("prop", "run-1:issue")
    assert stable_id("ev", "run-1:issue") != stable_id("ev", "run-2:issue")
