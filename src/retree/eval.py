from __future__ import annotations

from collections import Counter
from typing import Any

from .clients import LLMClient
from .types import RunResult
from .utils import normalize_text


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    if not gold_answers:
        return False
    normalized = normalize_text(prediction)
    return any(normalized == normalize_text(answer) for answer in gold_answers)


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(_token_f1_one(prediction, answer) for answer in gold_answers)


def evaluate_run(result: RunResult, gold_answers: list[str], judge: LLMClient | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "exact_match": exact_match(result.answer, gold_answers),
        "token_f1": token_f1(result.answer, gold_answers),
        "searches_used": result.metadata.get("searches_used", 0),
        "max_context_chars": result.metadata.get("max_context_chars", 0),
        "memory_conflict_detected_count": result.metadata.get("memory_conflict_detected_count", 0),
        "memory_repair_applied_count": result.metadata.get("memory_repair_applied_count", 0),
        "memory_revision_event_count": result.metadata.get("memory_revision_event_count", 0),
        "memory_pruned_node_count": result.metadata.get("memory_pruned_node_count", 0),
    }
    if judge is not None and gold_answers:
        metrics["llm_judge_failed"] = 0
        try:
            metrics["llm_judge_correct"] = judge_answer_correctness(judge, result.question, result.answer, gold_answers)
        except Exception as exc:  # Judge is optional; keep the main experiment result.
            metrics["llm_judge_failed"] = 1
            metrics["llm_judge_error"] = _compact_exception(exc)
    if result.claims and result.evidence:
        metrics.update(citation_diagnostics(result))
        if judge is not None:
            metrics["citation_support_failed"] = 0
            try:
                metrics.update(judge_citation_support(judge, result))
            except Exception as exc:  # Judge is optional; keep citation proxy metrics.
                metrics["citation_support_failed"] = 1
                metrics["citation_support_error"] = _compact_exception(exc)
    return metrics


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    aggregate: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values and all(isinstance(value, bool) for value in values):
            aggregate[key] = sum(1 for value in values if value) / len(values)
        elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            aggregate[key] = sum(values) / len(values)
    return aggregate


def judge_answer_correctness(judge: LLMClient, question: str, prediction: str, gold_answers: list[str]) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are evaluating QA correctness. Accept semantically equivalent answers. "
                'Return JSON only: {"correct":true,"reason":"..."}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nGold answers: {gold_answers}\nPrediction: {prediction}",
        },
    ]
    parsed = judge.chat_json(messages, fallback={"correct": False})
    return parsed.get("correct") is True


def citation_diagnostics(result: RunResult) -> dict[str, Any]:
    evidence_ids = {item.id for item in result.evidence}
    cited = []
    for claim in result.claims:
        ids = claim.get("evidence_ids", [])
        if isinstance(ids, list):
            cited.extend(str(item) for item in ids)
    if not cited:
        return {"citation_recall_proxy": 0.0, "citation_precision_proxy": 0.0}
    valid = [item for item in cited if item in evidence_ids]
    return {
        "citation_precision_proxy": len(valid) / len(cited),
        "citation_recall_proxy": len(set(valid)) / max(1, len(evidence_ids)),
    }


def judge_citation_support(judge: LLMClient, result: RunResult) -> dict[str, Any]:
    evidence_by_id = {item.id: item for item in result.evidence}
    judged = 0
    supported = 0
    for claim in result.claims:
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("claim", "")).strip()
        evidence_ids = claim.get("evidence_ids", [])
        if not claim_text or not isinstance(evidence_ids, list):
            continue
        cited = [evidence_by_id[str(item)] for item in evidence_ids if str(item) in evidence_by_id]
        if not cited:
            continue
        evidence_text = "\n".join(f"{item.id}: {item.text}\nsource={item.url}" for item in cited)
        messages = [
            {
                "role": "system",
                "content": (
                    "Judge whether the claim is fully supported by the cited evidence. "
                    'Return JSON only: {"supported":true,"reason":"..."}'
                ),
            },
            {"role": "user", "content": f"Question: {result.question}\nClaim: {claim_text}\nCited evidence:\n{evidence_text}"},
        ]
        parsed = judge.chat_json(messages, fallback={"supported": False})
        judged += 1
        supported += 1 if parsed.get("supported") is True else 0
    return {"citation_support_judged": judged, "citation_support_precision": supported / judged if judged else 0.0}


def _token_f1_one(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _compact_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:500]
