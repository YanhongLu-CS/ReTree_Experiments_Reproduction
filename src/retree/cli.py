from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback before dependencies are installed.
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from .agents import AgentConfig, SearchAgent
from .clients import DryRunLLMClient, DryRunSearchClient, LLMClient, SearchClient
from .data import load_examples
from .eval import aggregate_metrics, evaluate_run
from .types import AgentName
from .utils import append_jsonl, ensure_dir, load_yaml_config, write_json


AGENTS: tuple[AgentName, ...] = ("retree", "flat_update", "report_memory", "full_react")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ReTree reproduction experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one or more agents on a dataset.")
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--dataset", default="")
    run_parser.add_argument("--dataset-name", default="sample")
    run_parser.add_argument("--agents", nargs="+", choices=AGENTS, default=["retree"])
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--output-dir", default="")
    run_parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock LLM/search clients.")
    run_parser.add_argument("--judge", action="store_true", help="Use configured judge model for semantic correctness.")
    run_parser.set_defaults(func=run_command)

    args = parser.parse_args(argv)
    args.func(args)


def run_command(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    experiment_cfg = config.get("experiment", {})
    dataset_path = args.dataset or _dataset_path_from_config(config, args.dataset_name)
    output_dir = _make_output_dir(args.output_dir or experiment_cfg.get("output_dir", "runs"), args.dataset_name)
    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    llm = DryRunLLMClient() if args.dry_run else _make_llm(config.get("llm", {}))
    search = DryRunSearchClient() if args.dry_run else _make_search(config.get("search", {}))
    judge = None
    if args.judge or config.get("judge", {}).get("enabled") is True:
        judge = DryRunLLMClient() if args.dry_run else _make_llm(config.get("judge", {}))

    agent_cfg = AgentConfig(
        max_searches=int(experiment_cfg.get("max_searches", 8)),
        passages_per_search=int(experiment_cfg.get("passages_per_search", 5)),
        summary_word_limit=int(experiment_cfg.get("summary_word_limit", 140)),
        report_word_limit=int(experiment_cfg.get("report_word_limit", 200)),
        evidence_budget=int(experiment_cfg.get("evidence_budget", 5)),
        max_extracted_facts=int(experiment_cfg.get("max_extracted_facts", 6)),
        min_retrievals_before_stop=int(experiment_cfg.get("min_retrievals_before_stop", 1)),
    )
    examples = load_examples(dataset_path, limit=args.limit, seed=int(experiment_cfg.get("seed", 0)))
    per_agent_metrics: dict[str, list[dict[str, Any]]] = {agent: [] for agent in args.agents}

    if prediction_path.exists():
        prediction_path.unlink()

    for agent_name in args.agents:
        agent = SearchAgent(agent_name, llm, search, agent_cfg)
        for example in tqdm(examples, desc=f"{agent_name}"):
            result = agent.run(example)
            result.metrics = evaluate_run(result, example.answers, judge)
            per_agent_metrics[agent_name].append(result.metrics)
            append_jsonl(prediction_path, result.to_jsonable())

    aggregate = {agent: aggregate_metrics(rows) for agent, rows in per_agent_metrics.items()}
    write_json(metrics_path, {"output_dir": str(output_dir), "dataset": str(dataset_path), "agents": args.agents, "aggregate": aggregate})
    print(f"Wrote predictions to {prediction_path}")
    print(f"Wrote metrics to {metrics_path}")


def _dataset_path_from_config(config: dict[str, Any], name: str) -> str:
    datasets = config.get("experiment", {}).get("datasets", {})
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found in config experiment.datasets.")
    return str(datasets[name]["path"])


def _make_output_dir(root: str, dataset_name: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(root) / f"{dataset_name}_{stamp}")


def _make_llm(config: dict[str, Any]) -> LLMClient:
    return LLMClient(
        base_url=str(config.get("base_url", "")),
        api_key=str(config.get("api_key", "")),
        model=str(config.get("model", "")),
        temperature=float(config.get("temperature", 0)),
        max_tokens=int(config.get("max_tokens", 1200)),
        timeout_seconds=int(config.get("timeout_seconds", 90)),
    )


def _make_search(config: dict[str, Any]) -> SearchClient:
    return SearchClient(
        provider=str(config.get("provider", "custom")),
        base_url=str(config.get("base_url", "")),
        api_key=str(config.get("api_key", "")),
        timeout_seconds=int(config.get("timeout_seconds", 45)),
        max_passage_chars=int(config.get("max_passage_chars", 1200)),
    )


if __name__ == "__main__":
    main()
