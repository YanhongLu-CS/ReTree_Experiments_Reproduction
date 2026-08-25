from retree.clients import parse_search_response


def test_parse_serper_style_response() -> None:
    data = {"organic": [{"title": "A", "link": "https://example.com", "snippet": "Useful snippet."}]}
    passages = parse_search_response(data, "query", 5, 1200)
    assert len(passages) == 1
    assert passages[0].url == "https://example.com"
    assert passages[0].text == "Useful snippet."
