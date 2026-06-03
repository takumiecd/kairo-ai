"""Convert mixed Japanese/ASCII text into IME-style romaji input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pykakasi
from sudachipy import dictionary
from sudachipy import tokenizer


INPUT_PUNCTUATION = {
    "、": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
}


def is_japanese_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3040 <= codepoint <= 0x309F  # Hiragana
        or 0x30A0 <= codepoint <= 0x30FF  # Katakana
        or 0x3400 <= codepoint <= 0x4DBF  # CJK extension A
        or 0x4E00 <= codepoint <= 0x9FFF  # CJK unified ideographs
    )


@dataclass(frozen=True)
class TextSpan:
    kind: str
    text: str


@dataclass(frozen=True)
class ConvertedSpan:
    kind: str
    source: str
    input: str


def split_mixed_text(text: str) -> list[TextSpan]:
    """Split text into Japanese and non-Japanese spans."""
    spans: list[TextSpan] = []
    current: list[str] = []
    current_kind: str | None = None

    for char in text:
        kind = "japanese" if is_japanese_char(char) else "literal"
        if current_kind is not None and kind != current_kind:
            spans.append(TextSpan(current_kind, "".join(current)))
            current = []
        current_kind = kind
        current.append(char)

    if current:
        spans.append(TextSpan(current_kind or "literal", "".join(current)))

    return spans


class JapaneseRomajiConverter:
    """Convert Japanese spans to romaji while preserving literal spans."""

    def __init__(self) -> None:
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.kks = pykakasi.kakasi()

    def convert_text(self, text: str) -> str:
        return "".join(span.input for span in self.convert_spans(text))

    def convert_spans(self, text: str) -> list[ConvertedSpan]:
        parts: list[str] = []
        for span in split_mixed_text(text):
            if span.kind == "japanese":
                parts.append(
                    ConvertedSpan(
                        kind=span.kind,
                        source=span.text,
                        input=self.convert_japanese_span(span.text),
                    )
                )
            else:
                parts.append(
                    ConvertedSpan(
                        kind=span.kind,
                        source=span.text,
                        input=normalize_literal_span(span.text),
                    )
                )
        return parts

    def convert_japanese_span(self, text: str) -> str:
        romaji_parts: list[str] = []
        for token in self.tokenizer_obj.tokenize(text, self.mode):
            reading = token.reading_form()
            target_text = reading if reading and reading != "*" else token.surface()
            for item in self.kks.convert(target_text):
                romaji_parts.append(item["hepburn"])
        return "".join(romaji_parts)


def normalize_literal_span(text: str) -> str:
    return "".join(INPUT_PUNCTUATION.get(char, char) for char in text)


def convert_many(converter: JapaneseRomajiConverter, texts: Iterable[str]) -> list[str]:
    return [converter.convert_text(text) for text in texts]
