"""Shared dataset/vocab helpers used by every model family."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from dataset.vocab import CharVocab
from dataset.vocab import build_input_vocab
from dataset.vocab import build_output_bpe_vocab
from dataset.vocab import build_output_vocab


@dataclass(frozen=True)
class TrainingVocabs:
    input_vocab: CharVocab
    output_vocab: CharVocab

    @property
    def input_pad_id(self) -> int:
        return self.input_vocab.token_to_id["<pad>"]

    @property
    def output_pad_id(self) -> int:
        return self.output_vocab.token_to_id["<pad>"]

    @property
    def blank_id(self) -> int:
        return self.output_vocab.token_to_id["<blank>"]

    @property
    def bos_id(self) -> int:
        return self.output_vocab.token_to_id["<bos>"]


def load_jsonl_examples(path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def build_vocabs_from_records(
    records: list[dict[str, str]],
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
    vocab_sample: int | None = None,
    vocab_sample_seed: int = 0,
) -> TrainingVocabs:
    targets = [record["target"] for record in records]
    if output_tokenizer == "char":
        output_vocab = build_output_vocab(targets)
    elif output_tokenizer == "bpe":
        output_vocab = build_output_bpe_vocab(
            targets,
            vocab_size=output_vocab_size,
            min_frequency=output_min_token_frequency,
            sample_size=vocab_sample,
            seed=vocab_sample_seed,
        )
    else:
        raise ValueError("output_tokenizer must be one of: char, bpe")

    return TrainingVocabs(
        input_vocab=build_input_vocab([record["input"] for record in records]),
        output_vocab=output_vocab,
    )
