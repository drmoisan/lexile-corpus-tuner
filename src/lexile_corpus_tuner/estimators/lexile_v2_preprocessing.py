from __future__ import annotations

import pickle
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Tuple, cast

_ALPHA_PATTERN = re.compile(r"[A-Za-z]")
_nltk_cache: (
    tuple[
        type[Any],
        type[Any],
        Callable[..., list[str]],
        Callable[..., list[str]],
    ]
    | None
) = None


def load_stopwords(path: str | Path) -> list[str]:
    """Load the serialized stopword list shipped with lexile-determination-v2."""
    data = _load_pickle(path)
    if isinstance(data, (list, tuple, set)):
        sequence = cast(Iterable[Any], data)
        return [str(item) for item in sequence]
    if isinstance(data, dict):
        keys_iterable = cast(Iterable[Any], data.keys())
        return [str(key) for key in keys_iterable]
    if isinstance(data, str):
        return data.splitlines()
    raise TypeError(f"Unsupported stopword payload type: {type(data)!r}")


def vectorize_with_lexile_pipeline(
    text: str, tokenizer: Any, stopwords: Sequence[str]
) -> Any:
    """Replicate the upstream preprocessing + vectorization pipeline.

    This mirrors lexile-determination-v2/src/app.py, which:
    1. Splits the text into topical segments with TextTiling.
    2. Lemmatizes tokens with WordNet.
    3. Converts the lemmas to TF-IDF via the serialized Keras tokenizer.
    """
    lemmas = _lemmatize_segments(_segment_text(text, stopwords))
    return tokenizer.texts_to_matrix([lemmas], mode="tfidf")


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _segment_text(text: str, stopwords: Sequence[str]) -> list[str]:
    _, TextTilingTokenizer, _, _ = _ensure_nltk_dependencies()
    tokenizer = TextTilingTokenizer(stopwords=list(stopwords))
    try:
        raw_segments = tokenizer.tokenize(text)
    except ValueError:
        raw_segments = [text]
    if isinstance(raw_segments, tuple):
        # TextTilingTokenizer may return a tuple; the first element contains the segments.
        candidates_iterable = cast(Iterable[Any], raw_segments[0])
    elif isinstance(raw_segments, str):
        return [raw_segments]
    else:
        candidates_iterable = cast(Iterable[Any], raw_segments)
    return [str(segment) for segment in candidates_iterable]


def _lemmatize_segments(segments: Iterable[str]) -> list[str]:
    WordNetLemmatizer, _, sent_tokenize, word_tokenize = _ensure_nltk_dependencies()
    lemmatizer = WordNetLemmatizer()
    combined = " ".join(str(segment) for segment in segments)
    tokens: list[str] = []
    for sentence in sent_tokenize(combined):
        tokens.extend(word_tokenize(sentence))
    lemmas: list[str] = []
    for token in tokens:
        if _ALPHA_PATTERN.search(token):
            lemmas.append(lemmatizer.lemmatize(token))
        else:
            lemmas.append(token)
    return lemmas


def _ensure_nltk_dependencies() -> Tuple[
    type[Any],
    type[Any],
    Callable[..., list[str]],
    Callable[..., list[str]],
]:
    global _nltk_cache
    if _nltk_cache is None:
        try:
            stem_module = import_module("nltk.stem")
            tokenize_module = import_module("nltk.tokenize")
        except ModuleNotFoundError as exc:  # pragma: no cover - informative
            raise ImportError(
                "nltk is required for the lexile_v2 preprocessing pipeline. "
                "Install the 'lexile-v2' extra (e.g., `pip install .[lexile-v2]`)."
            ) from exc
        WordNetLemmatizer = getattr(stem_module, "WordNetLemmatizer")
        TextTilingTokenizer = getattr(tokenize_module, "TextTilingTokenizer")
        sent_tokenize = getattr(tokenize_module, "sent_tokenize")
        word_tokenize = getattr(tokenize_module, "word_tokenize")
        _nltk_cache = (
            WordNetLemmatizer,
            TextTilingTokenizer,
            sent_tokenize,
            word_tokenize,
        )
    return _nltk_cache
