"""Character vocabulary helpers."""

from __future__ import annotations

from dataclasses import dataclass


INPUT_SPECIAL_TOKENS = ["<pad>", "<unk>"]
OUTPUT_SPECIAL_TOKENS = ["<pad>", "<blank>", "<bos>", "<unk>"]


@dataclass(frozen=True)
class CharVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    unk_token: str = "<unk>"

    def encode(self, text: str) -> list[int]:
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(char, unk_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[index] for index in ids)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)


def build_input_vocab(texts: list[str]) -> CharVocab:
    return build_char_vocab(texts, INPUT_SPECIAL_TOKENS)


def build_output_vocab(texts: list[str]) -> CharVocab:
    return build_char_vocab(texts, OUTPUT_SPECIAL_TOKENS)


def build_char_vocab(texts: list[str], special_tokens: list[str]) -> CharVocab:
    chars = sorted({char for text in texts for char in text})
    id_to_token = list(special_tokens) + [
        char for char in chars if char not in special_tokens
    ]
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return CharVocab(token_to_id=token_to_id, id_to_token=id_to_token)
