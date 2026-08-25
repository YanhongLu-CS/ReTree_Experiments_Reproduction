# ReTree Reproduction Harness

This project implements an experiment harness for reproducing the paper:

> ReTree: Self-Correcting Long-Horizon Search Agents via Tree-Structured Memory  
> arXiv:2608.10676

The code is designed for third-party relay APIs. Fill `.env` with your LLM/search relay keys, then run the same agent grid used in the paper:

- `retree`: tree-structured memory with conflict-triggered evidence replacement and descendant pruning.
- `flat_update`: flat summary plus top-k evidence, with in-place contradiction updates.
- `report_memory`: compact running report plus visited URLs.
- `full_react`: full raw search trajectory.

The default reproduction settings mirror the paper-level setup: max 8 searches, 5 passages per search, 140-word ReTree/FlatUpdate summaries, top-5 evidence, and a 200-word report baseline.

## Setup

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,datasets]"
```

Edit `.env`:

```bash
LLM_BASE_URL=https://your-relay.example.com/v1
LLM_API_KEY=sk-your-key-here
LLM_MODEL=qwen3-8b

SEARCH_PROVIDER=custom
SEARCH_BASE_URL=https://your-search-relay.example.com/search
SEARCH_API_KEY=search-key-here

JUDGE_BASE_URL=https://your-relay.example.com/v1
JUDGE_API_KEY=sk-your-judge-key-here
JUDGE_MODEL=gpt-5
```

The LLM endpoint should be OpenAI-compatible at:

```text
POST {LLM_BASE_URL}/chat/completions
```

For search, built-in providers are `custom`, `serper`, `serpapi`, `tavily`, and `mock`. `custom` sends:

```json
{"query":"...","q":"...","top_k":5,"num":5}
```

and accepts common response shapes like `results`, `passages`, `organic`, `items`, or Bing-style `webPages.value`.

## Smoke Test

Run without real API keys:

```bash
retree run --config configs/default.yaml --dataset-name sample --agents retree flat_update report_memory full_react --dry-run
pytest
```

## Dataset Format

Normalize Bamboogle, HotpotQA, 2WikiMultiHopQA, and FRAMES into JSONL:

```json
{"id":"example_id","question":"...","answers":["gold answer","alias"],"metadata":{}}
```

Use the converter for local files:

```bash
python scripts/prepare_datasets.py local \
  --input raw/hotpotqa_validation.json \
  --output data/hotpotqa.jsonl \
  --question-field question \
  --answer-field answer \
  --id-field _id
```

Or Hugging Face datasets:

```bash
python scripts/prepare_datasets.py hf \
  --dataset-id hotpot_qa \
  --split validation \
  --output data/hotpotqa.jsonl \
  --question-field question \
  --answer-field answer \
  --id-field id
```

Add paths to `configs/default.yaml`:

```yaml
experiment:
  datasets:
    bamboogle:
      path: data/bamboogle.jsonl
    hotpotqa:
      path: data/hotpotqa.jsonl
    twowiki:
      path: data/2wiki.jsonl
    frames:
      path: data/frames.jsonl
```

## Run Experiments

One dataset:

```bash
retree run \
  --config configs/default.yaml \
  --dataset-name hotpotqa \
  --agents retree flat_update report_memory full_react \
  --limit 600
```

With LLM judge scoring:

```bash
retree run \
  --config configs/default.yaml \
  --dataset-name frames \
  --agents retree flat_update report_memory full_react \
  --judge
```

Outputs are written under `runs/<dataset>_<timestamp>/`:

- `predictions.jsonl`: full trajectories, memory events, extracted evidence, final claims.
- `metrics.json`: aggregate EM, token-F1, searches used, context length, optional LLM judge correctness and citation support.

ReTree/FlatUpdate memory diagnostics are included in both per-example `metadata` and aggregate metrics:

- `memory_conflict_detected_count`: how many update steps detected a contradiction candidate.
- `memory_repair_applied_count`: how many contradiction candidates were confirmed and repaired.
- `memory_revision_event_count`: number of memory revision events.
- `memory_pruned_node_count`: number of ReTree descendant nodes pruned after repair.

## Notes For Faithful Reproduction

- Use the same model family for all agents in one run.
- Keep search budget fixed across agents.
- Use the same search provider and passage extraction behavior across agents.
- Third-party search APIs can change retrieved passages, so exact numbers may differ from the paper unless the original search backend and dataset samples are matched.
- `--judge` adds semantic correctness and citation-support checks, but also increases model-call cost.
