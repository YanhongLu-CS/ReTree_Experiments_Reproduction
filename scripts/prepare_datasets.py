#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert QA datasets to the JSONL format expected by retree.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local", help="Convert a local JSON/JSONL/CSV-like file with configurable fields.")
    local.add_argument("--input", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--question-field", default="question")
    local.add_argument("--answer-field", default="answer")
    local.add_argument("--id-field", default="id")
    local.set_defaults(func=convert_local)

    hf = subparsers.add_parser("hf", help="Download a Hugging Face dataset split and normalize it to JSONL.")
    hf.add_argument("--dataset-id", required=True)
    hf.add_argument("--split", required=True)
    hf.add_argument("--output", required=True)
    hf.add_argument("--question-field", default="question")
    hf.add_argument("--answer-field", default="answer")
    hf.add_argument("--id-field", default="id")
    hf.add_argument("--limit", type=int, default=None)
    hf.set_defaults(func=convert_hf)

    args = parser.parse_args()
    args.func(args)


def convert_local(args: argparse.Namespace) -> None:
    path = Path(args.input)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("data", [])
    elif path.suffix == ".csv":
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported input format: {path}")
    write_rows(rows, args.output, args.question_field, args.answer_field, args.id_field)


def convert_hf(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dataset support first: pip install -e .[datasets]") from exc

    dataset = load_dataset(args.dataset_id, split=args.split)
    rows = []
    for index, row in enumerate(dataset):
        rows.append(dict(row))
        if args.limit is not None and index + 1 >= args.limit:
            break
    write_rows(rows, args.output, args.question_field, args.answer_field, args.id_field)


def write_rows(rows: list[dict[str, Any]], output: str, question_field: str, answer_field: str, id_field: str) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            question = row.get(question_field)
            if not question:
                continue
            raw_answer = row.get(answer_field, "")
            answers = normalize_answers(raw_answer)
            item = {
                "id": str(row.get(id_field) or row.get("_id") or f"example_{index:05d}"),
                "question": str(question),
                "answers": answers,
                "metadata": {key: value for key, value in row.items() if key not in {question_field, answer_field}},
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {target}")


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        answers = []
        for key in ["answer", "value", "normalized_value"]:
            if value.get(key):
                answers.append(str(value[key]))
        if isinstance(value.get("aliases"), list):
            answers.extend(str(item) for item in value["aliases"])
        return answers
    return [part.strip() for part in str(value).split("|") if part.strip()]


if __name__ == "__main__":
    main()
