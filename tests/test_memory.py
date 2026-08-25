from retree.clients import DryRunLLMClient
from retree.memory import ReTreeMemory
from retree.types import Evidence, SearchPassage


def test_retree_appends_child_and_tracks_active_evidence() -> None:
    memory = ReTreeMemory()
    passage = SearchPassage(id="p1", query="q", title="T", url="u", text="fact", rank=1)
    fact = Evidence(id="new_0", text="Amsterdam is in the Netherlands.", passage_id="p1", url="u", title="T", created_step=1)

    event, accepted = memory.integrate("Where is Amsterdam?", 1, [fact], [passage], DryRunLLMClient())

    assert event["type"] == "append_child"
    assert accepted[0].id == "e1"
    assert memory.all_evidence()[0].text.startswith("Amsterdam")
