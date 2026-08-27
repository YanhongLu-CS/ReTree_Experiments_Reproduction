from __future__ import annotations

import re
from heapq import nlargest
from dataclasses import dataclass, field
from typing import Any, Protocol

from .clients import LLMClient
from .prompts import (
    confirm_revision_prompt,
    conflict_prompt,
    report_prompt,
    structured_summary_prompt,
)
from .types import Evidence, SearchPassage, StructuredSummary
from .utils import normalize_text, token_overlap_score, truncate_words


class AgentMemory(Protocol):
    def render_context(self, question: str) -> str:
        ...

    def evidence_context(self, question: str) -> str:
        ...

    def integrate(
        self,
        question: str,
        query: str,
        step: int,
        new_facts: list[Evidence],
        passages: list[SearchPassage],
        llm: LLMClient,
        action: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Evidence]]:
        ...

    def all_evidence(self) -> list[Evidence]:
        ...


@dataclass
class MemoryNode:
    id: str
    parent_id: str | None
    summary: StructuredSummary = field(default_factory=StructuredSummary)
    evidence_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    revision_history: list[dict[str, Any]] = field(default_factory=list)


class ReTreeMemory:
    def __init__(self, evidence_budget: int = 5, summary_word_limit: int = 140) -> None:
        self.evidence_budget = evidence_budget
        self.summary_word_limit = summary_word_limit
        self.nodes: dict[str, MemoryNode] = {"n0": MemoryNode(id="n0", parent_id=None)}
        self.active_node_id = "n0"
        self.evidence_by_id: dict[str, Evidence] = {}
        self.evidence_counter = 0
        self.node_counter = 0

    def render_context(self, question: str) -> str:
        node = self.nodes[self.active_node_id]
        summary = node.summary.render(self.summary_word_limit)
        top = self._top_evidence(f"{question} {summary}", self._active_path_evidence())
        evidence = "\n".join(_render_evidence(item) for item in top) or "No explicit evidence yet."
        return (
            f"Active branch: {self.active_node_id}\n"
            f"Summary: {summary}\n"
            f"Top evidence:\n{evidence}"
        )

    def evidence_context(self, question: str) -> str:
        evidence = "\n".join(_render_evidence(item) for item in self._active_path_evidence())
        return f"Evidence:\n{evidence or 'No evidence.'}"

    def all_evidence(self) -> list[Evidence]:
        return list(self.evidence_by_id.values())

    def integrate(
        self,
        question: str,
        query: str,
        step: int,
        new_facts: list[Evidence],
        passages: list[SearchPassage],
        llm: LLMClient,
        action: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Evidence]]:
        if not new_facts:
            return {"type": "noop", "reason": "no_extracted_facts"}, []

        retrieved_context = _retrieved_context(question, query, passages)
        existing = self._top_evidence(retrieved_context, self._active_path_evidence())
        proposed = llm.chat_json(conflict_prompt(question, existing, new_facts), fallback={"conflict": False}) if existing else {"conflict": False}
        conflict_event: dict[str, Any] = {
            "conflict_detected": proposed.get("conflict") is True,
            "repair_applied": False,
            "pruned_node_count": 0,
        }
        if proposed.get("conflict") is True:
            old_id = str(proposed.get("refuted_evidence_id", ""))
            replacement_index = int(proposed.get("replacement_fact_index", -1))
            old = self.evidence_by_id.get(old_id)
            if old is not None and 0 <= replacement_index < len(new_facts):
                node = self._find_introducer_node(old.id)
                replacement = new_facts[replacement_index]
                if not _same_scope_repair_candidate(old, replacement):
                    conflict_event.update(
                        {
                            "conflict_candidate_evidence_id": old.id,
                            "conflict_rejected_reason": "different_entity_attribute_or_scope",
                        }
                    )
                else:
                    confirm = llm.chat_json(
                        confirm_revision_prompt(
                            question,
                            node.summary.render(self.summary_word_limit) if node else "",
                            old.history,
                            old,
                            new_facts,
                            replacement_index,
                        ),
                        fallback={"confirm": False},
                    )
                    if confirm.get("confirm") is True and node is not None:
                        previous = old.__dict__.copy()
                        _replace_evidence_in_place(old, replacement, step, str(confirm.get("reason", "")), preserve_created_step=True)
                        node.revision_history.append(
                            {
                                "step": step,
                                "evidence_id": old.id,
                                "x_old": previous.get("text", ""),
                                "x_new": replacement.text,
                                "old_source": {
                                    "passage_id": previous.get("passage_id", ""),
                                    "title": previous.get("title", ""),
                                    "url": previous.get("url", ""),
                                },
                                "new_source": {
                                    "passage_id": replacement.passage_id,
                                    "title": replacement.title,
                                    "url": replacement.url,
                                },
                                "reason": str(confirm.get("reason", "")),
                            }
                        )
                        self._regenerate_node_summary(question, node.id, llm, include_prior=False)
                        pruned = self._prune_descendants(node.id)
                        self.active_node_id = node.id
                        return {
                            "type": "revision",
                            "conflict_detected": True,
                            "repair_applied": True,
                            "replaced_evidence_id": old.id,
                            "node_id": node.id,
                            "pruned_nodes": pruned,
                            "pruned_node_count": len(pruned),
                            "reason": confirm.get("reason", ""),
                        }, [old]
                    conflict_event.update(
                        {
                            "conflict_candidate_evidence_id": old.id,
                            "conflict_rejected_reason": str(confirm.get("reason", "")),
                        }
                    )
            else:
                conflict_event.update(
                    {
                        "conflict_rejected_reason": "invalid_conflict_reference",
                        "conflict_candidate_evidence_id": old_id,
                    }
                )

        child_id = self._new_node_id()
        parent = self.nodes[self.active_node_id]
        child = MemoryNode(id=child_id, parent_id=parent.id)
        self.nodes[child_id] = child
        parent.child_ids.append(child_id)
        accepted = [self._assign_evidence_id(fact, child_id, step) for fact in new_facts]
        child.evidence_ids.extend(item.id for item in accepted)
        self.active_node_id = child_id
        self._regenerate_node_summary(question, child_id, llm)
        return {
            "type": "append_child",
            **conflict_event,
            "node_id": child_id,
            "accepted_evidence_ids": [item.id for item in accepted],
        }, accepted

    def _new_node_id(self) -> str:
        self.node_counter += 1
        return f"n{self.node_counter}"

    def _next_evidence_id(self) -> str:
        self.evidence_counter += 1
        return f"e{self.evidence_counter}"

    def _assign_evidence_id(self, fact: Evidence, node_id: str, step: int) -> Evidence:
        fact.id = self._next_evidence_id()
        fact.introduced_node_id = node_id
        fact.created_step = step
        self.evidence_by_id[fact.id] = fact
        return fact

    def _regenerate_node_summary(self, question: str, node_id: str, llm: LLMClient, *, include_prior: bool = True) -> None:
        node = self.nodes[node_id]
        evidence = self._path_evidence(node_id)
        prior = self.nodes[node.parent_id].summary.render(self.summary_word_limit) if include_prior and node.parent_id else ""
        parsed = llm.chat_json(structured_summary_prompt(question, prior, evidence, self.summary_word_limit), fallback={})
        node.summary = _summary_from_payload(parsed, self.summary_word_limit)

    def _prune_descendants(self, node_id: str) -> list[str]:
        pruned: list[str] = []
        node = self.nodes[node_id]
        stack = list(node.child_ids)
        node.child_ids = []
        while stack:
            child_id = stack.pop()
            child = self.nodes.pop(child_id, None)
            if child is None:
                continue
            pruned.append(child_id)
            stack.extend(child.child_ids)
        return pruned

    def _find_introducer_node(self, evidence_id: str) -> MemoryNode | None:
        stack = ["n0"]
        while stack:
            node_id = stack.pop()
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if evidence_id in node.evidence_ids:
                return node
            stack.extend(node.child_ids)
        return None

    def _active_path_nodes(self) -> list[MemoryNode]:
        node_ids: list[str] = []
        current: str | None = self.active_node_id
        while current is not None:
            node_ids.append(current)
            current = self.nodes[current].parent_id
        return [self.nodes[node_id] for node_id in reversed(node_ids)]

    def _path_evidence(self, node_id: str) -> list[Evidence]:
        node_ids: list[str] = []
        current: str | None = node_id
        while current is not None:
            node_ids.append(current)
            current = self.nodes[current].parent_id
        evidence: list[Evidence] = []
        for current_id in reversed(node_ids):
            node = self.nodes[current_id]
            evidence.extend(self.evidence_by_id[evidence_id] for evidence_id in node.evidence_ids if evidence_id in self.evidence_by_id)
        return evidence

    def _active_path_evidence(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for node in self._active_path_nodes():
            evidence.extend(self.evidence_by_id[evidence_id] for evidence_id in node.evidence_ids if evidence_id in self.evidence_by_id)
        return evidence

    def _top_evidence(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        return nlargest(self.evidence_budget, evidence, key=lambda item: token_overlap_score(question, item.text))


class FlatUpdateMemory:
    def __init__(self, evidence_budget: int = 5, summary_word_limit: int = 140) -> None:
        self.evidence_budget = evidence_budget
        self.summary_word_limit = summary_word_limit
        self.summary = StructuredSummary()
        self.evidence: list[Evidence] = []
        self.evidence_by_id: dict[str, Evidence] = {}
        self.revision_history: list[dict[str, Any]] = []
        self.evidence_counter = 0

    def render_context(self, question: str) -> str:
        summary = self.summary.render(self.summary_word_limit)
        evidence = "\n".join(_render_evidence(item) for item in self._top_evidence(f"{question} {summary}"))
        return f"Summary: {summary}\nTop evidence:\n{evidence or 'No explicit evidence yet.'}"

    def evidence_context(self, question: str) -> str:
        evidence = "\n".join(_render_evidence(item) for item in self.evidence)
        return f"Evidence:\n{evidence or 'No evidence.'}"

    def all_evidence(self) -> list[Evidence]:
        return list(self.evidence)

    def integrate(
        self,
        question: str,
        query: str,
        step: int,
        new_facts: list[Evidence],
        passages: list[SearchPassage],
        llm: LLMClient,
        action: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Evidence]]:
        if not new_facts:
            return {"type": "noop", "reason": "no_extracted_facts"}, []

        existing = self._top_evidence(_retrieved_context(question, query, passages))
        proposed = llm.chat_json(conflict_prompt(question, existing, new_facts), fallback={"conflict": False}) if existing else {"conflict": False}
        conflict_event: dict[str, Any] = {
            "conflict_detected": proposed.get("conflict") is True,
            "repair_applied": False,
            "pruned_node_count": 0,
        }
        if proposed.get("conflict") is True:
            old_id = str(proposed.get("refuted_evidence_id", ""))
            replacement_index = int(proposed.get("replacement_fact_index", -1))
            old = self.evidence_by_id.get(old_id)
            if old is not None and 0 <= replacement_index < len(new_facts):
                replacement = new_facts[replacement_index]
                if not _same_scope_repair_candidate(old, replacement):
                    conflict_event.update(
                        {
                            "conflict_candidate_evidence_id": old.id,
                            "conflict_rejected_reason": "different_entity_attribute_or_scope",
                        }
                    )
                else:
                    confirm = llm.chat_json(
                        confirm_revision_prompt(question, self.summary.render(self.summary_word_limit), old.history, old, new_facts, replacement_index),
                        fallback={"confirm": False},
                    )
                    if confirm.get("confirm") is True:
                        previous = old.__dict__.copy()
                        _replace_evidence_in_place(old, replacement, step, str(confirm.get("reason", "")))
                        self.revision_history.append(
                            {
                                "step": step,
                                "evidence_id": old.id,
                                "previous": previous,
                                "replacement": replacement.__dict__.copy(),
                                "reason": str(confirm.get("reason", "")),
                            }
                        )
                        self._regenerate_summary(question, llm)
                        return {
                            "type": "flat_revision",
                            "conflict_detected": True,
                            "repair_applied": True,
                            "pruned_node_count": 0,
                            "replaced_evidence_id": old.id,
                            "reason": confirm.get("reason", ""),
                        }, [old]
                    conflict_event.update(
                        {
                            "conflict_candidate_evidence_id": old.id,
                            "conflict_rejected_reason": str(confirm.get("reason", "")),
                        }
                    )
            else:
                conflict_event.update(
                    {
                        "conflict_rejected_reason": "invalid_conflict_reference",
                        "conflict_candidate_evidence_id": old_id,
                    }
                )

        accepted = [self._assign_evidence_id(fact, step) for fact in new_facts]
        self.evidence.extend(accepted)
        self._regenerate_summary(question, llm)
        return {"type": "flat_append", **conflict_event, "accepted_evidence_ids": [item.id for item in accepted]}, accepted

    def _assign_evidence_id(self, fact: Evidence, step: int) -> Evidence:
        self.evidence_counter += 1
        fact.id = f"e{self.evidence_counter}"
        fact.created_step = step
        fact.introduced_node_id = "flat"
        self.evidence_by_id[fact.id] = fact
        return fact

    def _regenerate_summary(self, question: str, llm: LLMClient) -> None:
        prior = self.summary.render(self.summary_word_limit)
        parsed = llm.chat_json(structured_summary_prompt(question, prior, self.evidence, self.summary_word_limit), fallback={})
        self.summary = _summary_from_payload(parsed, self.summary_word_limit)

    def _top_evidence(self, question: str) -> list[Evidence]:
        return nlargest(self.evidence_budget, self.evidence, key=lambda item: token_overlap_score(question, item.text))


class ReportMemory:
    def __init__(self, report_word_limit: int = 200) -> None:
        self.report_word_limit = report_word_limit
        self.report = ""
        self.visited: list[SearchPassage] = []

    def render_context(self, question: str) -> str:
        urls = "\n".join(f"- {url}" for url in self._url_bag())
        return f"Report: {self.report or 'No report yet.'}\nURL bag:\n{urls or 'None'}"

    def evidence_context(self, question: str) -> str:
        urls = "\n".join(f"- {url}" for url in self._url_bag())
        return f"Report: {self.report or 'No report yet.'}\nURL bag:\n{urls or 'None'}"

    def all_evidence(self) -> list[Evidence]:
        return []

    def integrate(
        self,
        question: str,
        query: str,
        step: int,
        new_facts: list[Evidence],
        passages: list[SearchPassage],
        llm: LLMClient,
        action: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Evidence]]:
        self.visited.extend(passages)
        parsed = llm.chat_json(report_prompt(question, self.report, passages, self.report_word_limit), fallback={})
        self.report = truncate_words(str(parsed.get("summary", "")).strip(), self.report_word_limit)
        return {
            "type": "report_update",
            "conflict_detected": False,
            "repair_applied": False,
            "pruned_node_count": 0,
            "visited_count": len(self.visited),
        }, []

    def _url_bag(self) -> list[str]:
        return sorted({passage.url for passage in self.visited if passage.url})


class FullTrajectoryMemory:
    def __init__(self) -> None:
        self.entries: list[str] = []
        self.evidence: list[Evidence] = []
        self.evidence_counter = 0

    def render_context(self, question: str) -> str:
        return "\n\n".join(self.entries) or "No trajectory yet."

    def evidence_context(self, question: str) -> str:
        evidence = "\n".join(_render_evidence(item) for item in self.evidence)
        transcript = "\n\n".join(self.entries)
        return f"Full trajectory:\n{transcript or 'No trajectory.'}\n\nExtracted evidence:\n{evidence or 'No explicit evidence.'}"

    def all_evidence(self) -> list[Evidence]:
        return list(self.evidence)

    def integrate(
        self,
        question: str,
        query: str,
        step: int,
        new_facts: list[Evidence],
        passages: list[SearchPassage],
        llm: LLMClient,
        action: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Evidence]]:
        accepted = []
        for fact in new_facts:
            self.evidence_counter += 1
            fact.id = f"e{self.evidence_counter}"
            fact.created_step = step
            fact.introduced_node_id = "trajectory"
            accepted.append(fact)
        self.evidence.extend(accepted)
        rendered = "\n".join(f"[{passage.rank}] {passage.title} {passage.url}\n{passage.raw_text or passage.text}" for passage in passages)
        rationale = ""
        if isinstance(action, dict):
            rationale = str(action.get("rationale", "")).strip()
        thought = f"Thought: {rationale}\n" if rationale else ""
        self.entries.append(f"Step {step}\n{thought}Action: search\nQuery: {query}\nObservation:\n{rendered}")
        return {
            "type": "trajectory_append",
            "conflict_detected": False,
            "repair_applied": False,
            "pruned_node_count": 0,
            "accepted_evidence_ids": [item.id for item in accepted],
        }, accepted


def make_memory(agent: str, evidence_budget: int, summary_word_limit: int, report_word_limit: int) -> AgentMemory:
    if agent == "retree":
        return ReTreeMemory(evidence_budget=evidence_budget, summary_word_limit=summary_word_limit)
    if agent == "flat_update":
        return FlatUpdateMemory(evidence_budget=evidence_budget, summary_word_limit=summary_word_limit)
    if agent == "report_memory":
        return ReportMemory(report_word_limit=report_word_limit)
    if agent == "full_react":
        return FullTrajectoryMemory()
    raise ValueError(f"Unknown agent: {agent}")


def _replace_evidence_in_place(
    old: Evidence,
    replacement: Evidence,
    step: int,
    reason: str,
    *,
    preserve_created_step: bool = False,
) -> None:
    old.history.append(
        {
            "step": step,
            "previous_text": old.text,
            "previous_url": old.url,
            "replacement_text": replacement.text,
            "replacement_url": replacement.url,
            "reason": reason,
        }
    )
    old.text = replacement.text
    old.passage_id = replacement.passage_id
    old.url = replacement.url
    old.title = replacement.title
    old.entity = replacement.entity
    old.attribute = replacement.attribute
    old.scope = replacement.scope
    if not preserve_created_step:
        old.created_step = step
    old.revision_count += 1


def _retrieved_context(question: str, query: str, passages: list[SearchPassage]) -> str:
    parts = [question, query]
    parts.extend(f"{passage.title} {passage.text}" for passage in passages)
    return " ".join(part for part in parts if part).strip()


def _same_scope_repair_candidate(old: Evidence, replacement: Evidence) -> bool:
    for field_name in ("entity", "attribute", "scope"):
        old_tokens = _slot_tokens(getattr(old, field_name))
        new_tokens = _slot_tokens(getattr(replacement, field_name))
        if old_tokens and new_tokens and old_tokens.isdisjoint(new_tokens):
            return False
    return True


def _slot_tokens(value: str) -> set[str]:
    return set(normalize_text(value).split())


def _render_evidence(evidence: Evidence) -> str:
    slot = " ".join(part for part in [evidence.entity, evidence.attribute, evidence.scope] if part)
    slot = f" ({slot})" if slot else ""
    return f"{evidence.id}{slot}: {evidence.text} [source: {evidence.title} {evidence.url}]"


def _summary_from_payload(payload: dict[str, Any], word_limit: int) -> StructuredSummary:
    summary = StructuredSummary(
        answer_slot=_clean_summary_text(_coerce_text(payload.get("answer_slot") or payload.get("summary", ""))),
        resolved_slots=_clean_summary_list(payload.get("resolved_slots")),
        open_slots=_clean_summary_list(payload.get("open_slots")),
        unresolved_candidates=_clean_summary_list(payload.get("unresolved_candidates")),
    )
    if not summary.answer_slot and not summary.resolved_slots and not summary.open_slots and not summary.unresolved_candidates:
        summary.open_slots = ["requested answer"]
    return _bound_structured_summary(summary, word_limit)


def _bound_structured_summary(summary: StructuredSummary, word_limit: int) -> StructuredSummary:
    while len(summary.render(word_limit * 10).split()) > word_limit:
        if summary.unresolved_candidates:
            summary.unresolved_candidates.pop()
        elif summary.open_slots:
            summary.open_slots.pop()
        elif summary.resolved_slots:
            summary.resolved_slots.pop()
        else:
            summary.answer_slot = truncate_words(summary.answer_slot, max(1, word_limit // 4))
            break
    return summary


def _clean_summary_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    cleaned: list[str] = []
    for item in values:
        text = _clean_summary_text(_coerce_text(item))
        if text:
            cleaned.append(text)
    return cleaned


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_coerce_text(item)}" for key, item in value.items() if item not in (None, "", []))
    if isinstance(value, list):
        return ", ".join(_coerce_text(item) for item in value)
    return str(value)


def _clean_summary_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b(?:evidence|passage|source)?\s*e\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:passage|source)\s*[:#]?\s*\w+://\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;,")
