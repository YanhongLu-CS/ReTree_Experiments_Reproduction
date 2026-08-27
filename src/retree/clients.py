from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - handled when real network clients are used.
    httpx = None  # type: ignore[assignment]

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover - dry-run works without tenacity.
    def retry(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator

    def stop_after_attempt(*args, **kwargs):  # type: ignore[no-redef]
        return None

    def wait_exponential(*args, **kwargs):  # type: ignore[no-redef]
        return None

from .types import SearchPassage
from .utils import extract_json_object, safe_get, truncate_words


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        timeout_seconds: int = 90,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        if not self.base_url or not self.api_key:
            raise RuntimeError("LLM base_url/api_key is empty. Fill .env or run with --dry-run.")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if httpx is None:
            raise RuntimeError("Install dependencies with `pip install -e .` before using real LLM APIs.")
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict[str, str]], *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = self.chat(messages)
        parsed = extract_json_object(raw)
        return parsed if parsed else (fallback or {})


class DryRunLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(base_url="", api_key="", model="dry-run")

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        prompt = "\n".join(message["content"] for message in messages)
        if '"action"' in prompt and '"search"' in prompt:
            question = _after_label(prompt, "Question:")
            step = _after_label(prompt, "Step:")
            if step.startswith("1"):
                return json.dumps(
                    {
                        "action": "search",
                        "query": question,
                        "answer_contract": {
                            "requested_value_type": "short answer",
                            "resolved_intermediate_slots": [],
                            "missing_relations": ["retrieved support"],
                            "final_slot_supported": False,
                        },
                        "rationale": "Dry-run first retrieval.",
                    }
                )
            return json.dumps(
                {
                    "action": "stop",
                    "answer": "Dry-run answer from gathered evidence.",
                    "answer_contract": {
                        "requested_value_type": "short answer",
                        "resolved_intermediate_slots": ["mock retrieval"],
                        "missing_relations": [],
                        "final_slot_supported": True,
                    },
                }
            )
        if '"facts"' in prompt and "Passage" in prompt:
            first = _after_label(prompt, "[0]")
            sentence = first.split(". ")[0].strip()
            if not sentence:
                sentence = "No usable fact extracted in dry run"
            return json.dumps({"facts": [{"passage_index": 0, "fact": sentence, "entity": "", "attribute": "", "scope": ""}]})
        if '"conflict"' in prompt:
            return json.dumps({"conflict": False, "refuted_evidence_id": "", "replacement_fact_index": -1, "reason": "Dry run does not revise."})
        if '"confirm"' in prompt:
            return json.dumps({"confirm": False, "reason": "Dry run does not revise."})
        if '"answer_slot"' in prompt:
            content = _after_label(prompt, "Evidence facts without IDs or URLs:")
            return json.dumps(
                {
                    "answer_slot": "requested answer",
                    "resolved_slots": [truncate_words(content.replace("\n", " "), 40)] if content else [],
                    "open_slots": [],
                    "unresolved_candidates": [],
                }
            )
        if '"summary"' in prompt:
            content = _after_label(prompt, "Evidence:")
            return json.dumps({"summary": truncate_words(content.replace("\n", " "), 120)})
        if '"supported"' in prompt:
            return json.dumps({"supported": True, "reason": "Dry-run support check passes after retrieval."})
        if '"answer"' in prompt and '"claims"' in prompt:
            evidence = _after_label(prompt, "Evidence:")
            claim = truncate_words(evidence.replace("\n", " "), 30) or "Dry-run answer from gathered evidence."
            return json.dumps({"answer": claim, "claims": [{"claim": claim, "evidence_ids": ["e1"]}]})
        return "{}"


def _after_label(text: str, label: str) -> str:
    marker = text.find(label)
    if marker == -1:
        return ""
    return text[marker + len(label) :].strip().split("\n", 1)[0].strip()


class SearchClient:
    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 45,
        max_passage_chars: int = 1200,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_passage_chars = max_passage_chars

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def search(self, query: str, top_k: int) -> list[SearchPassage]:
        provider = self.provider.lower()
        if provider == "mock":
            return _mock_search(query, top_k, self.max_passage_chars)
        if not self.api_key:
            raise RuntimeError("Search API key is empty. Fill .env or run with --dry-run.")

        if provider == "serper":
            data = self._post_json(
                self.base_url or "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key},
                payload={"q": query, "num": top_k},
            )
        elif provider == "tavily":
            data = self._post_json(
                self.base_url or "https://api.tavily.com/search",
                headers={},
                payload={"api_key": self.api_key, "query": query, "max_results": top_k, "include_answer": False},
            )
        elif provider == "serpapi":
            data = self._get_json(
                self.base_url or "https://serpapi.com/search.json",
                params={"q": query, "api_key": self.api_key, "num": top_k},
            )
        else:
            if not self.base_url:
                raise RuntimeError("SEARCH_BASE_URL is required for SEARCH_PROVIDER=custom.")
            data = self._post_json(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload={"query": query, "q": query, "top_k": top_k, "num": top_k},
            )

        return parse_search_response(data, query, top_k, self.max_passage_chars)

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        base_headers = {"Content-Type": "application/json", **headers}
        if httpx is None:
            raise RuntimeError("Install dependencies with `pip install -e .` before using real search APIs.")
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, headers=base_headers, json=payload)
            response.raise_for_status()
            return response.json()

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if httpx is None:
            raise RuntimeError("Install dependencies with `pip install -e .` before using real search APIs.")
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()


class DryRunSearchClient(SearchClient):
    def __init__(self) -> None:
        super().__init__(provider="mock", base_url="", api_key="")


def parse_search_response(data: Any, query: str, top_k: int, max_passage_chars: int) -> list[SearchPassage]:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = (
            data.get("passages")
            or data.get("results")
            or data.get("organic")
            or data.get("items")
            or data.get("webPages", {}).get("value")
            or []
        )
    else:
        rows = []

    passages: list[SearchPassage] = []
    for index, item in enumerate(rows[:top_k]):
        if not isinstance(item, dict):
            item = {"text": str(item)}
        title = str(safe_get(item, "title", "name", default=f"Result {index + 1}"))
        url = str(safe_get(item, "url", "link", "href", default=""))
        text = str(safe_get(item, "text", "content", "snippet", "body", "description", default=""))
        if not text and "paragraphs" in item and isinstance(item["paragraphs"], list):
            text = "\n".join(str(part) for part in item["paragraphs"])
        passages.append(
            SearchPassage(
                id=f"p{index + 1}",
                query=query,
                title=title,
                url=url,
                text=truncate_words(text.strip(), max(1, max_passage_chars // 5)),
                rank=index + 1,
                raw=item,
            )
        )
    return passages


def _mock_search(query: str, top_k: int, max_passage_chars: int) -> list[SearchPassage]:
    text = (
        "Mock retrieval passage for local smoke tests. "
        f"The query was: {query}. "
        "Replace this with a real search provider for paper reproduction."
    )
    return [
        SearchPassage(
            id=f"p{i + 1}",
            query=query,
            title=f"Mock result {i + 1}",
            url=f"mock://result/{i + 1}",
            text=truncate_words(text, max(1, max_passage_chars // 5)),
            rank=i + 1,
            raw={},
        )
        for i in range(top_k)
    ]
