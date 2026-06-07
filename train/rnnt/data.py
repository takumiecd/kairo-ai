"""Dataset and collation utilities for RNN-T training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from train.common.checkpoint import has_saved_vocab
from train.common.checkpoint import load_vocabs
from train.common.data import TrainingVocabs
from train.common.data import build_vocabs_from_records
from train.common.data import load_jsonl_examples


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    target_ids: list[int]
    input_text: str
    target_text: str


class JsonlTransducerDataset(Dataset):
    def __init__(self, examples: list[EncodedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def encode_records(
    records: list[dict[str, str]],
    vocabs: TrainingVocabs,
) -> JsonlTransducerDataset:
    return JsonlTransducerDataset(
        [
            EncodedExample(
                input_ids=vocabs.input_vocab.encode(record["input"]),
                target_ids=vocabs.output_vocab.encode(record["target"]),
                input_text=record["input"],
                target_text=record["target"],
            )
            for record in records
        ]
    )


def load_dataset_and_vocabs(path: Path) -> tuple[JsonlTransducerDataset, TrainingVocabs]:
    records = load_jsonl_examples(path)
    vocabs = build_vocabs_from_records(records)
    return encode_records(records, vocabs), vocabs


def load_train_valid_datasets_and_vocabs(
    train_path: Path,
    valid_path: Path | None,
    max_len: int | None = None,
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
    vocab_dir=None,
) -> tuple[JsonlTransducerDataset, JsonlTransducerDataset | None, TrainingVocabs]:
    train_records = load_jsonl_examples(train_path)
    valid_records = load_jsonl_examples(valid_path) if valid_path is not None else []
    if max_len is not None:
        train_records = [
            r for r in train_records
            if len(r["input"]) <= max_len and len(r["target"]) <= max_len
        ]
        valid_records = [
            r for r in valid_records
            if len(r["input"]) <= max_len and len(r["target"]) <= max_len
        ]
    if vocab_dir is not None and has_saved_vocab(vocab_dir):
        print(f"Reusing vocab from {vocab_dir}", flush=True)
        vocabs = load_vocabs(vocab_dir)
    else:
        vocabs = build_vocabs_from_records(
            train_records + valid_records,
            output_tokenizer=output_tokenizer,
            output_vocab_size=output_vocab_size,
            output_min_token_frequency=output_min_token_frequency,
        )
    train_dataset = encode_records(train_records, vocabs)
    valid_dataset = encode_records(valid_records, vocabs) if valid_path is not None else None
    return train_dataset, valid_dataset, vocabs


def collate_transducer_batch(
    examples: list[EncodedExample],
    vocabs: TrainingVocabs,
) -> dict[str, torch.Tensor]:
    batch_size = len(examples)
    input_lengths = torch.tensor(
        [len(example.input_ids) for example in examples],
        dtype=torch.int32,
    )
    target_lengths = torch.tensor(
        [len(example.target_ids) for example in examples],
        dtype=torch.int32,
    )
    max_input_len = int(input_lengths.max().item())
    max_target_len = int(target_lengths.max().item())

    inputs = torch.full(
        (batch_size, max_input_len),
        fill_value=vocabs.input_pad_id,
        dtype=torch.long,
    )
    targets = torch.full(
        (batch_size, max_target_len),
        fill_value=vocabs.output_pad_id,
        dtype=torch.int32,
    )
    prediction_inputs = torch.full(
        (batch_size, max_target_len + 1),
        fill_value=vocabs.output_pad_id,
        dtype=torch.long,
    )
    prediction_inputs[:, 0] = vocabs.bos_id

    for row, example in enumerate(examples):
        input_len = len(example.input_ids)
        target_len = len(example.target_ids)
        inputs[row, :input_len] = torch.tensor(example.input_ids, dtype=torch.long)
        targets[row, :target_len] = torch.tensor(example.target_ids, dtype=torch.int32)
        prediction_inputs[row, 1 : target_len + 1] = torch.tensor(
            example.target_ids,
            dtype=torch.long,
        )

    return {
        "inputs": inputs,
        "prediction_inputs": prediction_inputs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
    }
