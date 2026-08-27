from __future__ import annotations

import re

from .types import Evidence, SearchPassage


POLICY_SYSTEM = """You are a careful long-horizon web QA agent.
Return JSON only. Use either:
{"action":"search","query":"...","rationale":"..."}
or:
{"action":"stop","answer":"...","rationale":"..."}
Do not stop until the answer is supported by retrieved information."""


def policy_prompt(question: str, step: int, max_searches: int, memory_context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": POLICY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Step: {step} of {max_searches}\n"
                "Current memory:\n"
                f"{memory_context}\n\n"
                'Return JSON with "action" equal to "search" or "stop".'
            ),
        },
    ]


def extract_facts_prompt(question: str, passages: list[SearchPassage], max_facts: int) -> list[dict[str, str]]:
    rendered = "\n\n".join(
        f"[{index}] title={passage.title}\nurl={passage.url}\ntext={passage.text}"
        for index, passage in enumerate(passages)
    )
    return [
        {
            "role": "system",
            "content": (
                "Extract compact, source-grounded facts that may help answer the question. "
                "Each fact must be directly supported by exactly one numbered passage. "
                "Return JSON only: "
                '{"facts":[{"passage_index":0,"fact":"...","entity":"...","attribute":"...","scope":"..."}]}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nMax facts: {max_facts}\nPassages:\n{rendered}",
        },
    ]


def conflict_prompt(question: str, existing: list[Evidence], new_facts: list[Evidence]) -> list[dict[str, str]]:
    old = "\n".join(_format_evidence(evidence) for evidence in existing) or "None"
    new = "\n".join(f"[{index}] {_format_evidence(evidence)}" for index, evidence in enumerate(new_facts)) or "None"
    return [
        {
            "role": "system",
            "content": (
                "Detect whether one new fact directly corrects or supersedes one existing fact "
                "for the same entity, attribute, time/scope, or answer slot. "
                "Do not mark merely complementary evidence as conflict. Return JSON only: "
                '{"conflict":false,"refuted_evidence_id":"","replacement_fact_index":-1,"reason":"..."}'
            ),
        },
        {"role": "user", "content": f"Question: {question}\nExisting evidence:\n{old}\n\nNew facts:\n{new}"},
    ]


def confirm_revision_prompt(
    question: str,
    branch_summary: str,
    revision_history: list[dict[str, str]],
    old_evidence: Evidence,
    new_evidence: Evidence,
) -> list[dict[str, str]]:
    history = "\n".join(str(item) for item in revision_history[-8:]) or "None"
    return [
        {
            "role": "system",
            "content": (
                "Decide whether the new evidence should replace the old evidence in memory. "
                "Confirm only if the old evidence is likely wrong, stale, or less specific for the question. "
                'Return JSON only: {"confirm":true,"reason":"..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Branch summary: {branch_summary}\n"
                f"Recent revision history: {history}\n"
                f"Old evidence: {_format_evidence(old_evidence)}\n"
                f"New evidence: {_format_evidence(new_evidence)}"
            ),
        },
    ]


def summary_prompt(question: str, prior_summary: str, evidence: list[Evidence], word_limit: int) -> list[dict[str, str]]:
    rendered = "\n".join(_format_evidence(item) for item in evidence) or "No evidence yet."
    return [
        {
            "role": "system",
            "content": (
                "Write an up-to-date working memory summary for a web QA search agent. "
                "Keep only facts useful for answering the question and preserve uncertainty. "
                f"Use at most {word_limit} words. "
                'Return JSON only: {"summary":"..."}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nPrior summary: {prior_summary}\nEvidence:\n{rendered}",
        },
    ]


def structured_summary_prompt(question: str, prior_summary: str, evidence: list[Evidence], word_limit: int) -> list[dict[str, str]]:
    rendered = "\n".join(_format_evidence_for_summary(item) for item in evidence) or "No evidence yet."
    return [
        {
            "role": "system",
            "content": (
                "Update a bounded structured memory summary for a long-horizon web QA agent. "
                "The summary must record the requested answer slot, resolved intermediate slots, "
                "remaining open slots, and unresolved candidates. "
                "Do not include evidence IDs, passage IDs, URLs, source titles, or raw evidence objects. "
                f"The rendered summary must fit in at most {word_limit} words. "
                "Return JSON only: "
                '{"answer_slot":"...","resolved_slots":["..."],"open_slots":["..."],"unresolved_candidates":["..."]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Prior structured summary:\n{prior_summary}\n"
                "Evidence facts without IDs or URLs:\n"
                f"{rendered}"
            ),
        },
    ]


def report_prompt(question: str, prior_report: str, passages: list[SearchPassage], word_limit: int) -> list[dict[str, str]]:
    rendered = "\n\n".join(f"{item.title}\n{item.url}\n{item.text}" for item in passages)
    return [
        {
            "role": "system",
            "content": (
                "Update the running report for answering the question using the new search results. "
                f"The report must be at most {word_limit} words and may mention source URLs compactly. "
                'Return JSON only: {"summary":"..."}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nPrior report: {prior_report}\nNew search results:\n{rendered}",
        },
    ]


def support_prompt(question: str, proposed_answer: str, evidence_context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge whether the answer is supported by the provided evidence. "
                'Return JSON only: {"supported":true,"reason":"..."}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nProposed answer: {proposed_answer}\nEvidence:\n{evidence_context}",
        },
    ]


def final_answer_prompt(question: str, evidence_context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using only the evidence. Be concise. "
                "Return JSON only with answer and claim-level citations: "
                '{"answer":"...","claims":[{"claim":"...","evidence_ids":["e1"]}]}'
            ),
        },
        {"role": "user", "content": f"Question: {question}\nEvidence:\n{evidence_context}"},
    ]


def _format_evidence(evidence: Evidence) -> str:
    source = evidence.url or evidence.title
    slot = " ".join(part for part in [evidence.entity, evidence.attribute, evidence.scope] if part)
    slot = f" slot={slot}" if slot else ""
    return f"{evidence.id}: {evidence.text}{slot} source={source}"


def _format_evidence_for_summary(evidence: Evidence) -> str:
    slot = " ".join(part for part in [evidence.entity, evidence.attribute, evidence.scope] if part)
    slot = f" slot={slot}" if slot else ""
    text = re.sub(r"https?://\S+|www\.\S+", "", evidence.text)
    text = re.sub(r"\b(?:evidence|passage|source)?\s*e\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return f"- fact: {text}{slot}"
