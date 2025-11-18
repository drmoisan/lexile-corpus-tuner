from __future__ import annotations

import re
from typing import List

from .models import Token

TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?", re.UNICODE)


def tokenize_words(text: str) -> List[Token]:
    """Tokenize text into word tokens with character offsets."""
    tokens: List[Token] = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(
            Token(text=match.group(), start_char=match.start(), end_char=match.end())
        )
    return tokens
