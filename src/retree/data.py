from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from .types import QuestionExample
from .utils import read_jsonl


def load_examples(path: str | Path, *, limit: int | None = None, seed: int = 0) -> list[QuestionExample]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = read_jsonl(path) if path.suffix.lower() == ".jsonl" else _read_json_array(path)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported dataset file type: {path}")

    examples = [_row_to_example(row, index) for index, row in enumerate(rows)]
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit] if limit is not None else examples


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ["data", "examples", "questions", "rows"]:
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    raise ValueError(f"Could not find an example list in {path}")


def _row_to_example(row: dict[str, Any], index: int) -> QuestionExample:
    question = str(row.get("question") or row.get("query") or row.get("input") or "").strip()
    if not question:
        raise ValueError(f"Dataset row {index} does not contain a question/query/input field.")

    raw_answers = row.get("answers", row.get("gold_answers", row.get("answer", row.get("target", ""))))
    if isinstance(raw_answers, list):
        answers = [str(item) for item in raw_answers]
    elif isinstance(raw_answers, str):
        answers = [part.strip() for part in raw_answers.split("|") if part.strip()]
    else:
        answers = [str(raw_answers)] if raw_answers else []

    example_id = str(row.get("id") or row.get("_id") or row.get("qid") or f"example_{index:05d}")
    metadata = {key: value for key, value in row.items() if key not in {"question", "query", "input", "answers", "gold_answers", "answer", "target"}}
    return QuestionExample(id=example_id, question=question, answers=answers, metadata=metadata)


def load_hf_dataset(dataset_id: str, *, split: str, question_field: str = "question", answer_field: str = "answer", limit: int | None = None) -> list[QuestionExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies with `pip install -e .[datasets]` to load Hugging Face datasets.") from exc

    dataset = load_dataset(dataset_id, split=split)
    examples: list[QuestionExample] = []
    for index, row in enumerate(dataset):
        raw_answer = row.get(answer_field, "")
        if isinstance(raw_answer, dict) and "aliases" in raw_answer:
            answers = [str(item) for item in raw_answer.get("aliases", [])]
            if raw_answer.get("value"):
                answers.insert(0, str(raw_answer["value"]))
        elif isinstance(raw_answer, list):
            answers = [str(item) for item in raw_answer]
        else:
            answers = [str(raw_answer)] if raw_answer else []
        examples.append(
            QuestionExample(
                id=str(row.get("id") or row.get("_id") or f"{dataset_id}_{index}"),
                question=str(row[question_field]),
                answers=answers,
                metadata=dict(row),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples
