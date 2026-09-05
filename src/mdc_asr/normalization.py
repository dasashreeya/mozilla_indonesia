"""MDC transcript normalization, matched to the official runtime scorer."""

import re
from typing import Match


BRACKETED = re.compile(r"\[[^\]]+\]")
UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
WORD_PAREN = re.compile(r"\(([^()]*)\)")
PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
COMMA = re.compile(",+")
SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
SENTENCE_END = re.compile("[!?]+")
MULTISPACE = re.compile("  +")


def _lowercase_sentence_initial(match: Match[str]) -> str:
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text: str) -> str:
    """Normalize one transcript exactly as the published MDC score script does."""

    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    return re.sub(MULTISPACE, " ", text)

