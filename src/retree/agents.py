from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .clients import LLMClient, SearchClient
from .memory import AgentMemory, make_memory
from .prompts import extract_facts_prompt, final_answer_prompt, policy_prompt, support_prompt
from .types import AgentName, AgentStep, Evidence, QuestionExample, RunResult, SearchPassage
from .utils import normalize_text


@dataclass
class AgentConfig:
    max_searches: int = 8
    passages_per_search: int = 5
    summary_word_limit: int = 140
    report_word_limit: int = 200
    evidence_budget: int = 5
    max_extracted_facts: int = 6
    min_retrievals_before_stop: int = 1
    support_check: bool = True


class SearchAgent:
    def __init__(self, name: AgentName, llm: LLMClient, search_client: SearchClient, config: AgentConfig) -> None:
        self.name = name
        self.llm = llm
        self.search_client = search_client
        self.config = config

    def run(self, example: QuestionExample) -> RunResult:
        memory = make_memory(
            self.name,
            evidence_budget=self.config.evidence_budget,
            summary_word_limit=self.config.summary_word_limit,
            report_word_limit=self.config.report_word_limit,
        )
        steps: list[AgentStep] = []
        searches_used = 0
        real_retrievals_used = 0
        max_context_chars = 0
        stop_reason = "budget_exhausted"
        proposed_answer = ""
        passage_limit = min(self.config.passages_per_search, 5)

        for step in range(1, self.config.max_searches + 1):
            context = memory.render_context(example.question)
            max_context_chars = max(max_context_chars, len(context))
            action = self._choose_action(example.question, step, context)
            if (
                action.get("action") == "stop"
                and real_retrievals_used >= self.config.min_retrievals_before_stop
                and str(action.get("answer", "")).strip()
                and self._contract_allows_stop(action)
            ):
                proposed_answer = str(action.get("answer", "")).strip()
                if self._supported(example.question, proposed_answer, memory.evidence_context(example.question)):
                    steps.append(AgentStep(step=step, action="stop", answer=proposed_answer))
                    stop_reason = "model_stop_supported"
                    break

            query = self._query_from_action(action, example.question, step, memory)
            passages = self.search_client.search(query, passage_limit)
            passages = self._namespace_passage_ids(example.id, step, passages)
            searches_used += 1
            if passages:
                real_retrievals_used += 1
            extracted = self._extract_evidence(example.question, step, passages)
            memory_event, accepted = memory.integrate(example.question, query, step, extracted, passages, self.llm)
            steps.append(
                AgentStep(
                    step=step,
                    action="search",
                    query=query,
                    passages=passages,
                    extracted_facts=accepted,
                    memory_event=memory_event,
                )
            )

        final_payload = self._final_answer(example.question, proposed_answer, memory.evidence_context(example.question))
        answer = str(final_payload.get("answer") or proposed_answer or "").strip()
        claims = final_payload.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        memory_stats = _memory_event_stats(steps)

        return RunResult(
            example_id=example.id,
            agent=self.name,
            question=example.question,
            answer=answer,
            claims=claims,
            steps=steps,
            evidence=memory.all_evidence(),
            metadata={
                "searches_used": searches_used,
                "real_retrievals_used": real_retrievals_used,
                "stop_reason": stop_reason,
                "max_context_chars": max_context_chars,
                "max_searches": self.config.max_searches,
                "passages_per_search": passage_limit,
                **memory_stats,
            },
        )

    def _choose_action(self, question: str, step: int, context: str) -> dict[str, Any]:
        parsed = self.llm.chat_json(policy_prompt(question, step, self.config.max_searches, context), fallback={})
        action = str(parsed.get("action", "search")).lower()
        if action not in {"search", "stop"}:
            parsed["action"] = "search"
        return parsed

    def _contract_allows_stop(self, action: dict[str, Any]) -> bool:
        contract = action.get("answer_contract")
        if not isinstance(contract, dict):
            return False
        if contract.get("final_slot_supported") is False:
            return False
        missing = contract.get("missing_relations")
        return not (isinstance(missing, list) and any(_is_real_missing_relation(item) for item in missing))

    def _query_from_action(self, action: dict[str, Any], question: str, step: int, memory: AgentMemory) -> str:
        query = str(action.get("query", "")).strip()
        if query:
            return query
        if step == 1:
            return question
        context = memory.render_context(question)
        context_words = normalize_text(context).split()
        suffix = " ".join(context_words[:12])
        return f"{question} {suffix}".strip()

    def _extract_evidence(self, question: str, step: int, passages: list[SearchPassage]) -> list[Evidence]:
        if not passages:
            return []
        fact_limit = min(self.config.max_extracted_facts, 6)
        parsed = self.llm.chat_json(extract_facts_prompt(question, passages, fact_limit), fallback={"facts": []})
        facts = parsed.get("facts", [])
        if not isinstance(facts, list):
            return []
        extracted: list[Evidence] = []
        for index, item in enumerate(facts[:fact_limit]):
            if not isinstance(item, dict):
                continue
            try:
                passage_index = int(item.get("passage_index", item.get("source_passage_index", item.get("source_index", 0))))
            except (TypeError, ValueError):
                passage_index = 0
            if passage_index < 0 or passage_index >= len(passages):
                continue
            text = str(item.get("fact") or item.get("text") or "").strip()
            if not text:
                continue
            passage = passages[passage_index]
            extracted.append(
                Evidence(
                    id=f"new_{index}",
                    text=text,
                    passage_id=passage.id,
                    url=passage.url,
                    title=passage.title,
                    created_step=step,
                    entity=str(item.get("entity", "")),
                    attribute=str(item.get("attribute", "")),
                    scope=str(item.get("scope", "")),
                )
            )
        return extracted

    def _supported(self, question: str, proposed_answer: str, evidence_context: str) -> bool:
        if not self.config.support_check:
            return True
        parsed = self.llm.chat_json(support_prompt(question, proposed_answer, evidence_context), fallback={"supported": False})
        return parsed.get("supported") is True

    def _final_answer(self, question: str, proposed_answer: str, evidence_context: str) -> dict[str, Any]:
        parsed = self.llm.chat_json(final_answer_prompt(question, evidence_context), fallback={})
        if parsed.get("answer"):
            return parsed
        if proposed_answer:
            return {"answer": proposed_answer, "claims": []}
        return {"answer": "", "claims": []}

    def _namespace_passage_ids(self, example_id: str, step: int, passages: list[SearchPassage]) -> list[SearchPassage]:
        namespace = hashlib.md5(f"{example_id}:{step}".encode("utf-8")).hexdigest()[:8]
        for index, passage in enumerate(passages):
            passage.id = f"{namespace}_p{index + 1}"
        return passages


def _memory_event_stats(steps: list[AgentStep]) -> dict[str, int]:
    conflict_detected_count = 0
    repair_applied_count = 0
    pruned_node_count = 0
    revision_event_count = 0

    for step in steps:
        event = step.memory_event or {}
        conflict_detected_count += 1 if event.get("conflict_detected") is True else 0
        repair_applied_count += 1 if event.get("repair_applied") is True else 0
        revision_event_count += 1 if event.get("type") in {"revision", "flat_revision"} else 0
        try:
            pruned_node_count += int(event.get("pruned_node_count", 0))
        except (TypeError, ValueError):
            pass

    return {
        "memory_conflict_detected_count": conflict_detected_count,
        "memory_repair_applied_count": repair_applied_count,
        "memory_revision_event_count": revision_event_count,
        "memory_pruned_node_count": pruned_node_count,
    }


def _is_real_missing_relation(value: Any) -> bool:
    text = normalize_text(str(value))
    return text not in {"", "none", "no", "n a", "na", "n/a", "not applicable"}
