from utils import select_and_merge_sentences


def test_select_and_merge_sentences_basic():
    # simple increasing lengths to force merges around threshold
    sentences = ["a" * 10, "b" * 20, "c" * 30, "d" * 40]
    out = select_and_merge_sentences(sentences, min_length_for_merge=25)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)
    # at least one merged chunk should be >= threshold
    assert any(len(x) >= 25 for x in out)
