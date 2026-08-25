from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AgentName = Literal["retree", "flat_update", "report_memory", "full_react"]


@dataclass
class QuestionExample:
    id: str
    question: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPassage:
    id: str
    query: str
    title: str
    url: str
    text: str
    rank: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    id: str
    text: str
    passage_id: str
    url: str
    title: str
    created_step: int
    entity: str = ""
    attribute: str = ""
    scope: str = ""
    introduced_node_id: str = ""
    revision_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExtractedFact:
    text: str
    passage_index: int
    entity: str = ""
    attribute: str = ""
    scope: str = ""


@dataclass
class AgentStep:
    step: int
    action: str
    query: str = ""
    answer: str = ""
    passages: list[SearchPassage] = field(default_factory=list)
    extracted_facts: list[Evidence] = field(default_factory=list)
    memory_event: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    example_id: str
    agent: AgentName
    question: str
    answer: str
    claims: list[dict[str, Any]]
    steps: list[AgentStep]
    evidence: list[Evidence]
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "agent": self.agent,
            "question": self.question,
            "answer": self.answer,
            "claims": self.claims,
            "steps": [
                {
                    "step": step.step,
                    "action": step.action,
                    "query": step.query,
                    "answer": step.answer,
                    "passages": [passage.__dict__ for passage in step.passages],
                    "extracted_facts": [fact.__dict__ for fact in step.extracted_facts],
                    "memory_event": step.memory_event,
                }
                for step in self.steps
            ],
            "evidence": [evidence.__dict__ for evidence in self.evidence],
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
