from retree.agents import AgentConfig, SearchAgent
from retree.clients import DryRunLLMClient, DryRunSearchClient, SearchClient
from retree.types import QuestionExample, SearchPassage


def test_agent_dry_run_completes() -> None:
    agent = SearchAgent(
        "retree",
        DryRunLLMClient(),
        DryRunSearchClient(),
        AgentConfig(max_searches=2, passages_per_search=2),
    )
    result = agent.run(QuestionExample(id="x", question="What is a smoke test?", answers=["smoke test"]))

    assert result.answer
    assert result.metadata["searches_used"] >= 1


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def chat_json(self, messages, *, fallback=None):
        if not self.outputs:
            return fallback or {}
        return self.outputs.pop(0)


class RecordingSearch(SearchClient):
    def __init__(self):
        super().__init__(provider="mock", base_url="", api_key="")
        self.requested_top_k: list[int] = []

    def search(self, query: str, top_k: int) -> list[SearchPassage]:
        self.requested_top_k.append(top_k)
        return [
            SearchPassage(
                id=f"p{index + 1}",
                query=query,
                title=f"Result {index + 1}",
                url=f"https://example.com/{index + 1}",
                text=f"Fact source {index + 1}.",
                rank=index + 1,
            )
            for index in range(top_k)
        ]


def test_agent_caps_passages_and_extracted_facts() -> None:
    llm = ScriptedLLM(
        [
            {"action": "search", "query": "oversized search"},
            {
                "facts": [
                    {"passage_index": 0, "fact": f"Atomic fact {index}.", "entity": "entity", "attribute": "attr", "scope": "scope"}
                    for index in range(10)
                ]
            },
            {"answer_slot": "answer", "resolved_slots": ["Atomic fact 0"], "open_slots": [], "unresolved_candidates": []},
            {
                "action": "stop",
                "answer": "answer",
                "answer_contract": {"final_slot_supported": True, "missing_relations": []},
            },
            {"supported": True},
            {"answer": "answer", "claims": []},
        ]
    )
    search = RecordingSearch()
    agent = SearchAgent(
        "retree",
        llm,
        search,
        AgentConfig(max_searches=3, passages_per_search=99, max_extracted_facts=99),
    )

    result = agent.run(QuestionExample(id="x", question="Question?", answers=["answer"]))

    assert search.requested_top_k == [5]
    assert len(result.steps[0].passages) == 5
    assert len(result.steps[0].extracted_facts) == 6
    assert result.metadata["passages_per_search"] == 5
