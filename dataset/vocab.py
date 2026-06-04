"""Vocabulary and tokenizer helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random


INPUT_SPECIAL_TOKENS = ["<pad>", "<unk>"]
OUTPUT_SPECIAL_TOKENS = ["<pad>", "<blank>", "<bos>", "<unk>"]


@dataclass(frozen=True)
class CharVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    unk_token: str = "<unk>"

    def encode(self, text: str) -> list[int]:
        unk_id = self.token_to_id[self.unk_token]
        if not any(len(token) > 1 for token in self.token_to_id if not token.startswith("<")):
            return [self.token_to_id.get(char, unk_id) for char in text]

        ids: list[int] = []
        index = 0
        max_token_len = max(len(token) for token in self.token_to_id)
        while index < len(text):
            token_id = None
            max_end = min(len(text), index + max_token_len)
            for end in range(max_end, index, -1):
                candidate = text[index:end]
                if candidate in self.token_to_id:
                    token_id = self.token_to_id[candidate]
                    index = end
                    break
            if token_id is None:
                token_id = unk_id
                index += 1
            ids.append(token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[index] for index in ids)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)


def build_input_vocab(texts: list[str]) -> CharVocab:
    return build_char_vocab(texts, INPUT_SPECIAL_TOKENS)


def build_output_vocab(texts: list[str]) -> CharVocab:
    return build_char_vocab(texts, OUTPUT_SPECIAL_TOKENS)


def build_output_bpe_vocab(
    texts: list[str],
    vocab_size: int,
    min_frequency: int = 2,
    sample_size: int | None = None,
    seed: int = 0,
) -> CharVocab:
    # ベース文字は全テキストから集める（カバレッジを落とさない）。
    id_to_token = list(OUTPUT_SPECIAL_TOKENS)
    chars = sorted({char for text in texts for char in text})
    id_to_token.extend(char for char in chars if char not in id_to_token)

    # マージ計算は重いので、必要ならサンプルしたコーパスで回す（(B)）。
    merge_texts = texts
    if sample_size is not None and sample_size < len(texts):
        merge_texts = random.Random(seed).sample(texts, sample_size)

    sequences: dict[tuple[str, ...], int] = {}
    for text in merge_texts:
        if not text:
            continue
        tokens = tuple(text)
        sequences[tokens] = sequences.get(tokens, 0) + 1

    try:
        from tqdm import tqdm
        pbar = tqdm(
            total=max(0, vocab_size - len(id_to_token)),
            desc="Building BPE merges",
            leave=False,
        )
    except ImportError:
        pbar = None

    while len(id_to_token) < vocab_size:
        pair_counts: dict[tuple[str, str], int] = {}
        for tokens, count in sequences.items():
            for left, right in zip(tokens, tokens[1:]):
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        if not pair_counts:
            break

        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0][0] + item[0][1]),
        )
        if best_count < min_frequency:
            break

        merged = best_pair[0] + best_pair[1]
        if merged in id_to_token:
            break
        id_to_token.append(merged)
        if pbar is not None:
            pbar.update(1)

        next_sequences: dict[tuple[str, ...], int] = {}
        for tokens, count in sequences.items():
            merged_tokens: list[str] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged_tokens.append(merged)
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            merged_tuple = tuple(merged_tokens)
            next_sequences[merged_tuple] = next_sequences.get(merged_tuple, 0) + count
        sequences = next_sequences

    if pbar is not None:
        pbar.close()

    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return CharVocab(token_to_id=token_to_id, id_to_token=id_to_token)


def build_char_vocab(texts: list[str], special_tokens: list[str]) -> CharVocab:
    chars = sorted({char for text in texts for char in text})
    id_to_token = list(special_tokens) + [
        char for char in chars if char not in special_tokens
    ]
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return CharVocab(token_to_id=token_to_id, id_to_token=id_to_token)


def vocab_from_token_to_id(token_to_id: dict[str, int]) -> CharVocab:
    id_to_token = [""] * len(token_to_id)
    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token
    return CharVocab(token_to_id=dict(token_to_id), id_to_token=id_to_token)
