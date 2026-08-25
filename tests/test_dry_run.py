from retree.agents import AgentConfig, SearchAgent
from retree.clients import DryRunLLMClient, DryRunSearchClient
from retree.types import QuestionExample


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
