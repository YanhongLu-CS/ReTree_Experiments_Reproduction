from retree.clients import DryRunLLMClient
from retree.memory import FlatUpdateMemory, FullTrajectoryMemory, ReTreeMemory, ReportMemory
from retree.prompts import extract_facts_prompt, structured_summary_prompt
from retree.types import Evidence, SearchPassage, StructuredSummary


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def chat_json(self, messages, *, fallback=None):
        self.messages.append(messages)
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


def test_retree_answer_context_excludes_summary() -> None:
    memory = ReTreeMemory()
    memory.nodes["n0"].summary = StructuredSummary(answer_slot="SECRET_SUMMARY_SLOT")
    memory.evidence_by_id["e1"] = Evidence("e1", "Visible active evidence.", "p1", "u1", "T1", 1, introduced_node_id="n0")
    memory.nodes["n0"].evidence_ids = ["e1"]

    context = memory.evidence_context("question")

    assert "Visible active evidence." in context
    assert "SECRET_SUMMARY_SLOT" not in context


def test_retree_conflict_topk_uses_question_plus_retrieved_content() -> None:
    memory = ReTreeMemory(evidence_budget=1)
    memory.evidence_by_id["e1"] = Evidence("e1", "alpha beta gamma delta", "p1", "u1", "T1", 1, introduced_node_id="n0")
    memory.evidence_by_id["e2"] = Evidence("e2", "omega", "p2", "u2", "T2", 1, introduced_node_id="n0")
    memory.nodes["n0"].evidence_ids = ["e1", "e2"]
    passage = SearchPassage(id="p3", query="omega", title="Source", url="u3", text="omega", rank=1)
    new_fact = Evidence("new_0", "new fact", "p3", "u3", "T3", 2)
    llm = ScriptedLLM(
        [
            {"conflict": False},
            {"answer_slot": "answer", "resolved_slots": ["new fact"], "open_slots": [], "unresolved_candidates": []},
        ]
    )

    memory.integrate("alpha beta gamma delta", "omega", 2, [new_fact], [passage], llm)
    conflict_prompt_text = "\n".join(message["content"] for message in llm.messages[0])

    assert "e1:" in conflict_prompt_text
    assert "e2:" not in conflict_prompt_text


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
    memory.evidence_by_id["e1"].introduced_node_id = "wrong_cached_node"

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


def test_report_memory_context_keeps_only_report_and_url_bag() -> None:
    memory = ReportMemory()
    passage = SearchPassage(
        id="p1",
        query="q",
        title="Source",
        url="https://example.com/source",
        text="SECRET_SNIPPET should not remain in report context.",
        rank=1,
    )
    memory.integrate("question", "query", 1, [], [passage], ScriptedLLM([{"summary": "Synthesized report."}]))

    context = memory.evidence_context("question")

    assert "Synthesized report." in context
    assert "https://example.com/source" in context
    assert "SECRET_SNIPPET" not in context


def test_flat_update_uses_structured_summary_and_top_five_budget() -> None:
    memory = FlatUpdateMemory(evidence_budget=1)
    memory.summary = StructuredSummary(answer_slot="bridge opening date")
    memory.evidence = [
        Evidence("e1", "alpha unrelated detail", "p1", "u1", "T1", 1),
        Evidence("e2", "bridge opened in 2011", "p2", "u2", "T2", 1),
    ]
    memory.evidence_by_id = {item.id: item for item in memory.evidence}

    context = memory.render_context("When?")

    assert "Answer slot: bridge opening date" in context
    assert "bridge opened in 2011" in context
    assert "alpha unrelated detail" not in context


def test_flat_update_revision_does_not_prune_or_locate_tree_state() -> None:
    memory = FlatUpdateMemory()
    old = Evidence("new_0", "Old value.", "p1", "old-url", "Old", 1, entity="same", attribute="attr", scope="scope")
    memory.integrate("question", "old query", 1, [old], [], ScriptedLLM([
        {"answer_slot": "answer", "resolved_slots": ["Old value"], "open_slots": [], "unresolved_candidates": []}
    ]))
    replacement = Evidence("new_0", "New value.", "p2", "new-url", "New", 2, entity="same", attribute="attr", scope="scope")
    event, accepted = memory.integrate("question", "new query", 2, [replacement], [], ScriptedLLM([
        {"conflict": True, "refuted_evidence_id": "e1", "replacement_fact_index": 0},
        {"confirm": True, "reason": "better supported"},
        {"answer_slot": "answer", "resolved_slots": ["New value"], "open_slots": [], "unresolved_candidates": []},
    ]))

    assert event["type"] == "flat_revision"
    assert event["pruned_node_count"] == 0
    assert accepted[0].id == "e1"
    assert memory.evidence[0].text == "New value."
    assert memory.evidence[0].introduced_node_id == "flat"


def test_full_trajectory_memory_keeps_react_style_raw_observations() -> None:
    memory = FullTrajectoryMemory()
    passage = SearchPassage(
        id="p1",
        query="q",
        title="Source",
        url="https://example.com/source",
        text="Truncated.",
        rank=1,
        raw_text="RAW_OBSERVATION_TEXT retained in the full trajectory.",
    )
    memory.integrate(
        "question",
        "query text",
        1,
        [],
        [passage],
        ScriptedLLM([]),
        action={"rationale": "Need source evidence."},
    )

    context = memory.render_context("question")

    assert "Thought: Need source evidence." in context
    assert "Action: search" in context
    assert "Query: query text" in context
    assert "RAW_OBSERVATION_TEXT" in context
