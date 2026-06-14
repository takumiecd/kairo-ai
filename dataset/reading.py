"""Convert mixed Japanese/ASCII text into IME-style romaji input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pykakasi
from sudachipy import dictionary
from sudachipy import tokenizer


# Kana whose IME (wāpuro-rōmaji) input differs from pykakasi's Hepburn output.
# pykakasi targets a transliteration scheme, not the keys a typist presses:
#   ー (long vowel) -> Hepburn doubles the vowel (ループ "ruupu"), but IME types
#                      the '-' key (ループ "ru-pu").
#   ヅ/づ           -> Hepburn "zu" collides with ず; IME types "du".
#   ヂ/ぢ           -> Hepburn "ji" collides with じ; IME types "di".
#   ・ (middle dot) -> left as a full-width char; IME types the '/' key.
# We splice these in around pykakasi-converted runs of ordinary kana.
SPECIAL_KANA_INPUT = {
    "ー": "-",
    "ヅ": "du",
    "づ": "du",
    "ヂ": "di",
    "ぢ": "di",
    "・": "/",
}


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
            romaji_parts.append(self._convert_reading(target_text))
        return "".join(romaji_parts)

    def _convert_reading(self, target_text: str) -> str:
        """Convert a kana reading to IME-style (wāpuro-rōmaji) input.

        pykakasi emits Hepburn, which diverges from the keys a typist presses
        for a handful of kana (see ``SPECIAL_KANA_INPUT``). We convert runs of
        ordinary kana with pykakasi and splice in the IME spelling for each
        special kana, so the dataset matches what users actually type.
        """
        parts: list[str] = []
        run: list[str] = []
        for char in target_text:
            if char in SPECIAL_KANA_INPUT:
                parts.append(self._hepburn("".join(run)))
                run.clear()
                parts.append(SPECIAL_KANA_INPUT[char])
            else:
                run.append(char)
        parts.append(self._hepburn("".join(run)))
        return "".join(parts)

    def _hepburn(self, text: str) -> str:
        return "".join(item["hepburn"] for item in self.kks.convert(text))


def normalize_literal_span(text: str) -> str:
    return "".join(INPUT_PUNCTUATION.get(char, char) for char in text)


def convert_many(converter: JapaneseRomajiConverter, texts: Iterable[str]) -> list[str]:
    return [converter.convert_text(text) for text in texts]
