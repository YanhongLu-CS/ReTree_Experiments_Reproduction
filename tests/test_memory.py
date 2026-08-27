from retree.clients import DryRunLLMClient
from retree.memory import ReTreeMemory
from retree.prompts import extract_facts_prompt, structured_summary_prompt
from retree.types import Evidence, SearchPassage, StructuredSummary


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def chat_json(self, messages, *, fallback=None):
        if not self.outputs:
            return fallback or {}
        return self.outputs.pop(0)


def test_retree_appends_child_and_tracks_active_evidence() -> None:
    memory = ReTreeMemory()
    passage = SearchPassage(id="p1", query="q", title="T", url="u", text="fact", rank=1)
    fact = Evidence(id="new_0", text="Amsterdam is in the Netherlands.", passage_id="p1", url="u", title="T", created_step=1)

    event, accepted = memory.integrate("Where is Amsterdam?", "Amsterdam location", 1, [fact], [passage], DryRunLLMClient())

    assert event["type"] == "append_child"
    assert accepted[0].id == "e1"
    assert memory.all_evidence()[0].text.startswith("Amsterdam")


def test_structured_summary_prompt_hides_evidence_ids_and_urls() -> None:
    evidence = Evidence(
        id="e99",
        text="Creme Puff won in 1999 according to https://example.com and evidence e12.",
        passage_id="p9",
        url="https://source.example/p9",
        title="Source Title",
        created_step=1,
        entity="Creme Puff",
        attribute="award year",
        scope="Cat of the Year",
    )

    messages = structured_summary_prompt("When was the award?", "none", [evidence], 140)
    rendered = "\n".join(message["content"] for message in messages)

    assert "e99" not in rendered
    assert "e12" not in rendered
    assert "https://example.com" not in rendered
    assert "https://source.example" not in rendered
    assert "Source Title" not in rendered


def test_extractor_binds_sources_by_passage_index_without_urls() -> None:
    passage = SearchPassage(
        id="p1",
        query="q",
        title="Indexed result",
        url="https://source.example/result",
        text="The answer-relevant fact appears here.",
        rank=1,
    )

    messages = extract_facts_prompt("What is the answer?", [passage], 6)
    rendered = "\n".join(message["content"] for message in messages)

    assert "passage_index" in rendered
    assert "https://source.example/result" not in rendered


def test_retree_topk_uses_question_plus_summary() -> None:
    memory = ReTreeMemory(evidence_budget=1)
    memory.nodes["n0"].summary = StructuredSummary(answer_slot="bridge opening date")
    memory.evidence_by_id["e1"] = Evidence("e1", "alpha unrelated detail", "p1", "u1", "T1", 1, introduced_node_id="n0")
    memory.evidence_by_id["e2"] = Evidence("e2", "bridge opened in 2011", "p2", "u2", "T2", 1, introduced_node_id="n0")
    memory.nodes["n0"].evidence_ids = ["e1", "e2"]

    context = memory.render_context("When?")

    assert "bridge opened in 2011" in context
    assert "alpha unrelated detail" not in context


def test_retree_revision_repairs_same_evidence_id_and_prunes_context_only() -> None:
    memory = ReTreeMemory()
    passage1 = SearchPassage(id="p1", query="q", title="Old", url="old-url", text="old", rank=1)
    old = Evidence(id="new_0", text="Old answer slot.", passage_id="p1", url="old-url", title="Old", created_step=1)
    first_llm = ScriptedLLM(
        [
            {
                "answer_slot": "answer unknown",
                "resolved_slots": ["old slot"],
                "open_slots": ["final answer"],
                "unresolved_candidates": [],
            }
        ]
    )
    memory.integrate("question", "old query", 1, [old], [passage1], first_llm)

    passage2 = SearchPassage(id="p2", query="q", title="Child", url="child-url", text="child", rank=1)
    child = Evidence(id="new_0", text="Child branch fact.", passage_id="p2", url="child-url", title="Child", created_step=2)
    second_llm = ScriptedLLM(
        [
            {"conflict": False},
            {
                "answer_slot": "answer unknown",
                "resolved_slots": ["old slot", "child branch fact"],
                "open_slots": ["final answer"],
                "unresolved_candidates": [],
            },
        ]
    )
    memory.integrate("question", "child query", 2, [child], [passage2], second_llm)

    passage3 = SearchPassage(id="p3", query="q", title="New", url="new-url", text="new", rank=1)
    replacement = Evidence(id="new_0", text="Corrected answer slot.", passage_id="p3", url="new-url", title="New", created_step=3)
    third_llm = ScriptedLLM(
        [
            {"conflict": True, "refuted_evidence_id": "e1", "replacement_fact_index": 0},
            {"confirm": True, "reason": "new fact corrects old fact"},
            {
                "answer_slot": "corrected answer slot",
                "resolved_slots": ["corrected slot"],
                "open_slots": [],
                "unresolved_candidates": [],
            },
        ]
    )
    event, accepted = memory.integrate("question", "corrected query", 3, [replacement], [passage3], third_llm)

    assert event["type"] == "revision"
    assert event["replaced_evidence_id"] == "e1"
    assert event["pruned_nodes"] == ["n2"]
    assert accepted[0].id == "e1"
    assert set(memory.evidence_by_id) == {"e1", "e2"}
    assert memory.evidence_by_id["e1"].url == "new-url"
    assert memory.evidence_by_id["e1"].text == "Corrected answer slot."
    assert memory.evidence_by_id["e1"].created_step == 1
    assert memory.evidence_by_id["e1"].history[0]["previous_url"] == "old-url"
    assert memory.evidence_by_id["e2"].url == "child-url"
    assert [item.id for item in memory._active_path_evidence()] == ["e1"]
