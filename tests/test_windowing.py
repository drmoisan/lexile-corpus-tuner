from lexile_corpus_tuner.models import Document
from lexile_corpus_tuner.tokenization import tokenize_words
from lexile_corpus_tuner.windowing import create_windows


def test_create_windows_overlaps_correctly():
    doc = Document(doc_id="test", text="one two three four five six")
    tokens = tokenize_words(doc.text)
    windows = create_windows(doc, tokens, window_size=4, stride=2)

    assert len(windows) == 2
    assert windows[0].text.strip().startswith("one")
    assert windows[1].text.strip().startswith("three")
    assert windows[0].start_token_idx == 0
    assert windows[1].start_token_idx == 2
    assert windows[-1].end_token_idx == len(tokens)
