from lexile_corpus_tuner.tokenization import tokenize_words


def test_tokenize_words_returns_offsets():
    text = "Hello, world! It's sunny today."
    tokens = tokenize_words(text)

    assert [token.text for token in tokens] == ["Hello", "world", "It's", "sunny", "today"]
    assert tokens[0].start_char == 0
    assert tokens[0].end_char == 5
    assert tokens[-1].text == "today"
    assert text[tokens[-1].start_char : tokens[-1].end_char] == "today"
