from retree.eval import exact_match, token_f1


def test_exact_match_normalizes_articles_and_case() -> None:
    assert exact_match("The Amsterdam.", ["Amsterdam"])


def test_token_f1_partial_overlap() -> None:
    assert 0 < token_f1("Python creator worked in Amsterdam", ["Amsterdam"]) < 1
