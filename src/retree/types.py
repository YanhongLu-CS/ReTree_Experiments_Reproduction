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
    raw_text: str = ""
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
class StructuredSummary:
    answer_slot: str = ""
    resolved_slots: list[str] = field(default_factory=list)
    open_slots: list[str] = field(default_factory=list)
    unresolved_candidates: list[str] = field(default_factory=list)

    def render(self, word_limit: int = 140) -> str:
        parts = [
            f"Answer slot: {self.answer_slot or 'unknown'}",
            f"Resolved slots: {_join_or_none(self.resolved_slots)}",
            f"Open slots: {_join_or_none(self.open_slots)}",
            f"Unresolved candidates: {_join_or_none(self.unresolved_candidates)}",
        ]
        return _truncate_words("; ".join(parts), word_limit)


def _join_or_none(values: list[str]) -> str:
    cleaned = [value.strip() for value in values if value.strip()]
    return " | ".join(cleaned) if cleaned else "none"


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


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
    passages: list[SearchPassage] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        evidence_map = {evidence.id: _evidence_json(evidence) for evidence in self.evidence}
        passage_map = {_passage_map_key(passage): _passage_json(passage) for passage in self.passages}
        return {
            "example_id": self.example_id,
            "agent": self.agent,
            "question": self.question,
            "answer": self.answer,
            "claims": self.claims,
            "passages": [_passage_json(passage) for passage in self.passages],
            "passage_map": passage_map,
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
            "evidence": [_evidence_json(evidence) for evidence in self.evidence],
            "evidence_map": evidence_map,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


def _passage_map_key(passage: SearchPassage) -> str:
    return passage.url or passage.id


def _passage_json(passage: SearchPassage) -> dict[str, Any]:
    return {
        "id": passage.id,
        "query": passage.query,
        "title": passage.title,
        "url": passage.url,
        "text": passage.text,
        "raw_text": passage.raw_text or passage.text,
        "rank": passage.rank,
    }


def _evidence_json(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "text": evidence.text,
        "passage_id": evidence.passage_id,
        "url": evidence.url,
        "title": evidence.title,
        "created_step": evidence.created_step,
        "entity": evidence.entity,
        "attribute": evidence.attribute,
        "scope": evidence.scope,
        "introduced_node_id": evidence.introduced_node_id,
        "revision_count": evidence.revision_count,
        "history": evidence.history,
    }
