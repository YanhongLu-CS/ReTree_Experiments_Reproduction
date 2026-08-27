from __future__ import annotations

from collections import Counter
from typing import Any

from .clients import LLMClient
from .types import RunResult, SearchPassage
from .utils import normalize_text, token_overlap_score


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
    elif result.claims and result.agent == "report_memory" and result.passages:
        metrics.update({"citation_recall_proxy": 0.0, "citation_precision_proxy": 0.0})
    if result.claims and (result.evidence or (result.agent == "report_memory" and result.passages)):
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
    passage_by_url = _passage_by_url(result.passages)
    if not evidence_by_id and result.agent == "report_memory":
        return _judge_report_posthoc_citation_support(judge, result)

    judged = 0
    entails = 0
    neutral = 0
    contradicts = 0
    missing_passages = 0
    for claim in result.claims:
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("claim", "")).strip()
        evidence_ids = claim.get("evidence_ids", [])
        if not claim_text or not isinstance(evidence_ids, list):
            continue
        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(str(evidence_id))
            if evidence is None:
                continue
            passage = passage_by_url.get(evidence.url)
            if passage is None:
                missing_passages += 1
                continue
            label = _judge_passage_nli(judge, result.question, claim_text, passage, evidence.id, evidence.text, evidence.url)
            judged += 1
            if label == "entails":
                entails += 1
            elif label == "contradicts":
                contradicts += 1
            else:
                neutral += 1
    precision = entails / judged if judged else 0.0
    return {
        "citation_support_judged": judged,
        "citation_support_precision": precision,
        "citation_entailment_precision": precision,
        "citation_entails_count": entails,
        "citation_neutral_count": neutral,
        "citation_contradicts_count": contradicts,
        "citation_missing_passage_count": missing_passages,
    }


def _judge_report_posthoc_citation_support(judge: LLMClient, result: RunResult) -> dict[str, Any]:
    judged_claims = 0
    passage_pairs_judged = 0
    entails = 0
    neutral = 0
    contradicts = 0
    reconstructed = 0
    for claim in result.claims:
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("claim", "")).strip()
        if not claim_text:
            continue
        candidates = sorted(
            result.passages,
            key=lambda passage: token_overlap_score(claim_text, passage.raw_text or passage.text),
            reverse=True,
        )[:5]
        if not candidates:
            continue
        judged_claims += 1
        labels = []
        for passage in candidates:
            labels.append(_judge_passage_nli(judge, result.question, claim_text, passage, "posthoc", "", passage.url))
            passage_pairs_judged += 1
        if "entails" in labels:
            entails += 1
            reconstructed += 1
        elif "contradicts" in labels:
            contradicts += 1
        else:
            neutral += 1
    precision = entails / judged_claims if judged_claims else 0.0
    return {
        "citation_support_judged": judged_claims,
        "citation_support_precision": precision,
        "citation_entailment_precision": precision,
        "citation_entails_count": entails,
        "citation_neutral_count": neutral,
        "citation_contradicts_count": contradicts,
        "citation_missing_passage_count": 0,
        "citation_posthoc_reconstructed_count": reconstructed,
        "citation_passage_pairs_judged": passage_pairs_judged,
    }


def _judge_passage_nli(
    judge: LLMClient,
    question: str,
    claim_text: str,
    passage: SearchPassage,
    evidence_id: str,
    evidence_text: str,
    evidence_url: str,
) -> str:
    passage_text = passage.raw_text or passage.text
    messages = [
        {
            "role": "system",
            "content": (
                "You are a passage-level NLI judge. Label whether the passage entails, "
                "is neutral toward, or contradicts the claim. Check the cited passage, "
                "not the claim or evidence text in isolation. Return JSON only: "
                '{"label":"entails","reason":"..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Claim: {claim_text}\n"
                f"Evidence identifier: {evidence_id}\n"
                f"Current evidence text: {evidence_text or 'N/A'}\n"
                f"Current evidence URL: {evidence_url or passage.url}\n"
                f"Passage URL: {passage.url}\n"
                f"Passage text:\n{passage_text}"
            ),
        },
    ]
    parsed = judge.chat_json(messages, fallback={"label": "neutral"})
    return _normalize_nli_label(parsed.get("label"))


def _passage_by_url(passages: list[SearchPassage]) -> dict[str, SearchPassage]:
    mapped: dict[str, SearchPassage] = {}
    for passage in passages:
        if passage.url and passage.url not in mapped:
            mapped[passage.url] = passage
    return mapped


def _normalize_nli_label(value: Any) -> str:
    text = normalize_text(str(value))
    if text in {"entails", "entail", "entailed", "support", "supports", "supported"}:
        return "entails"
    if text in {"contradicts", "contradict", "contradiction", "refutes", "refute"}:
        return "contradicts"
    return "neutral"


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
