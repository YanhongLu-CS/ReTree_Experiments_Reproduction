from retree.eval import exact_match, judge_citation_support, token_f1
from retree.types import Evidence, RunResult, SearchPassage


def test_exact_match_normalizes_articles_and_case() -> None:
    assert exact_match("The Amsterdam.", ["Amsterdam"])


def test_token_f1_partial_overlap() -> None:
    assert 0 < token_f1("Python creator worked in Amsterdam", ["Amsterdam"]) < 1


class CapturingJudge:
    def __init__(self, label="entails"):
        self.label = label
        self.messages = []

    def chat_json(self, messages, *, fallback=None):
        self.messages.append(messages)
        return {"label": self.label, "reason": "scripted"}


def test_citation_judge_checks_raw_passage_text_by_url() -> None:
    passage = SearchPassage(
        id="p1",
        query="q",
        title="Source",
        url="https://example.com/source",
        text="Truncated snippet.",
        rank=1,
        raw_text="RAW_PASSAGE_TEXT supports the atomic claim.",
    )
    evidence = Evidence(
        id="e1",
        text="Short extracted evidence.",
        passage_id="p1",
        url="https://example.com/source",
        title="Source",
        created_step=1,
    )
    result = RunResult(
        example_id="x",
        agent="retree",
        question="Question?",
        answer="Answer",
        claims=[{"claim": "Atomic claim", "evidence_ids": ["e1"]}],
        steps=[],
        evidence=[evidence],
        passages=[passage],
    )
    judge = CapturingJudge()

    metrics = judge_citation_support(judge, result)
    prompt = judge.messages[0][1]["content"]
    payload = result.to_jsonable()

    assert metrics["citation_entails_count"] == 1
    assert "RAW_PASSAGE_TEXT" in prompt
    assert "Passage URL: https://example.com/source" in prompt
    assert payload["evidence_map"]["e1"]["url"] == "https://example.com/source"
    assert payload["passage_map"]["https://example.com/source"]["raw_text"].startswith("RAW_PASSAGE_TEXT")


def test_report_baseline_posthoc_reconstructs_claim_source_links() -> None:
    passage = SearchPassage(
        id="p1",
        query="q",
        title="Source",
        url="https://example.com/source",
        text="Claim token.",
        rank=1,
        raw_text="Claim token appears in the original passage.",
    )
    result = RunResult(
        example_id="x",
        agent="report_memory",
        question="Question?",
        answer="Answer",
        claims=[{"claim": "Claim token", "evidence_ids": []}],
        steps=[],
        evidence=[],
        passages=[passage],
    )

    metrics = judge_citation_support(CapturingJudge(), result)

    assert metrics["citation_posthoc_reconstructed_count"] == 1
    assert metrics["citation_passage_pairs_judged"] == 1
