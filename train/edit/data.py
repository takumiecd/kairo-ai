"""Dataset utilities for neural edit transducer training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from train.common.data import TrainingVocabs
from train.common.data import build_vocabs_from_records
from train.common.data import load_jsonl_examples


KEEP = 0
DELETE = 1
INSERT = 2
STOP = 3
ACTION_BOS = 4
ACTION_PAD = -100
INSERT_PAD = -100

ACTION_NAMES = {
    KEEP: "KEEP",
    DELETE: "DELETE",
    INSERT: "INSERT",
    STOP: "STOP",
}


@dataclass(frozen=True)
class EditAction:
    op_id: int
    token_id: int | None = None


@dataclass(frozen=True)
class EncodedEditExample:
    input_ids: list[int]
    previous_ids: list[int]
    target_ids: list[int]
    actions: list[EditAction]
    input_text: str
    previous_text: str
    target_text: str


class JsonlEditDataset(Dataset):
    def __init__(self, examples: list[EncodedEditExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedEditExample:
        return self.examples[index]


def build_min_edit_script(
    previous_ids: list[int],
    target_ids: list[int],
) -> list[EditAction]:
    """Return a cursor-based minimum edit script from previous to target tokens."""
    prev_len = len(previous_ids)
    target_len = len(target_ids)
    costs = [[0] * (target_len + 1) for _ in range(prev_len + 1)]

    for i in range(prev_len - 1, -1, -1):
        costs[i][target_len] = costs[i + 1][target_len] + 1
    for j in range(target_len - 1, -1, -1):
        costs[prev_len][j] = costs[prev_len][j + 1] + 1

    for i in range(prev_len - 1, -1, -1):
        for j in range(target_len - 1, -1, -1):
            choices = []
            if previous_ids[i] == target_ids[j]:
                choices.append((costs[i + 1][j + 1], KEEP))
            choices.append((costs[i + 1][j] + 1, DELETE))
            choices.append((costs[i][j + 1] + 1, INSERT))
            costs[i][j] = min(cost for cost, _op in choices)

    actions: list[EditAction] = []
    i = 0
    j = 0
    while i < prev_len or j < target_len:
        if (
            i < prev_len
            and j < target_len
            and previous_ids[i] == target_ids[j]
            and costs[i][j] == costs[i + 1][j + 1]
        ):
            actions.append(EditAction(KEEP))
            i += 1
            j += 1
        elif i < prev_len and costs[i][j] == costs[i + 1][j] + 1:
            actions.append(EditAction(DELETE))
            i += 1
        else:
            actions.append(EditAction(INSERT, target_ids[j]))
            j += 1

    actions.append(EditAction(STOP))
    return actions


def apply_edit_script(previous_ids: list[int], actions: list[EditAction]) -> list[int]:
    output_ids: list[int] = []
    cursor = 0
    for action in actions:
        if action.op_id == KEEP:
            if cursor >= len(previous_ids):
                raise ValueError("KEEP cannot be applied past the previous token sequence")
            output_ids.append(previous_ids[cursor])
            cursor += 1
        elif action.op_id == DELETE:
            if cursor >= len(previous_ids):
                raise ValueError("DELETE cannot be applied past the previous token sequence")
            cursor += 1
        elif action.op_id == INSERT:
            if action.token_id is None:
                raise ValueError("INSERT requires token_id")
            output_ids.append(action.token_id)
        elif action.op_id == STOP:
            break
        else:
            raise ValueError(f"Unknown edit op id: {action.op_id}")
    return output_ids


def encode_edit_example(
    input_text: str,
    previous_text: str,
    target_text: str,
    vocabs: TrainingVocabs,
) -> EncodedEditExample:
    previous_ids = vocabs.output_vocab.encode(previous_text)
    target_ids = vocabs.output_vocab.encode(target_text)
    return EncodedEditExample(
        input_ids=vocabs.input_vocab.encode(input_text),
        previous_ids=previous_ids,
        target_ids=target_ids,
        actions=build_min_edit_script(previous_ids, target_ids),
        input_text=input_text,
        previous_text=previous_text,
        target_text=target_text,
    )


def encode_edit_records(
    records: list[dict[str, str]],
    vocabs: TrainingVocabs,
) -> JsonlEditDataset:
    return JsonlEditDataset(
        [
            encode_edit_example(
                input_text=record["input"],
                previous_text=record.get("previous", ""),
                target_text=record["target"],
                vocabs=vocabs,
            )
            for record in records
        ]
    )


def load_edit_dataset_and_vocabs(path) -> tuple[JsonlEditDataset, TrainingVocabs]:
    records = load_jsonl_examples(path)
    vocabs = build_edit_vocabs_from_records(records)
    return encode_edit_records(records, vocabs), vocabs


def build_edit_vocabs_from_records(
    records: list[dict[str, str]],
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
) -> TrainingVocabs:
    vocab_records = [
        {
            "input": record["input"],
            "target": record.get("previous", "") + record["target"],
        }
        for record in records
    ]
    return build_vocabs_from_records(
        vocab_records,
        output_tokenizer=output_tokenizer,
        output_vocab_size=output_vocab_size,
        output_min_token_frequency=output_min_token_frequency,
    )


def load_train_valid_edit_datasets_and_vocabs(
    train_path,
    valid_path=None,
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
) -> tuple[JsonlEditDataset, JsonlEditDataset | None, TrainingVocabs]:
    train_records = load_jsonl_examples(train_path)
    valid_records = load_jsonl_examples(valid_path) if valid_path is not None else []
    vocabs = build_edit_vocabs_from_records(
        train_records + valid_records,
        output_tokenizer=output_tokenizer,
        output_vocab_size=output_vocab_size,
        output_min_token_frequency=output_min_token_frequency,
    )
    train_dataset = encode_edit_records(train_records, vocabs)
    valid_dataset = encode_edit_records(valid_records, vocabs) if valid_path is not None else None
    return train_dataset, valid_dataset, vocabs


def collate_edit_batch(
    examples: list[EncodedEditExample],
    vocabs: TrainingVocabs,
) -> dict[str, torch.Tensor]:
    batch_size = len(examples)
    input_lengths = torch.tensor([len(example.input_ids) for example in examples], dtype=torch.int32)
    previous_lengths = torch.tensor(
        [len(example.previous_ids) for example in examples],
        dtype=torch.int32,
    )
    action_lengths = torch.tensor([len(example.actions) for example in examples], dtype=torch.int32)

    max_input_len = int(input_lengths.max().item()) if batch_size else 0
    max_previous_len = int(previous_lengths.max().item()) if batch_size else 0
    max_action_len = int(action_lengths.max().item()) if batch_size else 0

    inputs = torch.full(
        (batch_size, max_input_len),
        fill_value=vocabs.input_pad_id,
        dtype=torch.long,
    )
    previous_tokens = torch.full(
        (batch_size, max_previous_len),
        fill_value=vocabs.output_pad_id,
        dtype=torch.long,
    )
    action_input_ops = torch.full(
        (batch_size, max_action_len),
        fill_value=ACTION_BOS,
        dtype=torch.long,
    )
    action_input_insert_tokens = torch.full(
        (batch_size, max_action_len),
        fill_value=vocabs.output_pad_id,
        dtype=torch.long,
    )
    action_target_ops = torch.full(
        (batch_size, max_action_len),
        fill_value=ACTION_PAD,
        dtype=torch.long,
    )
    action_target_insert_tokens = torch.full(
        (batch_size, max_action_len),
        fill_value=INSERT_PAD,
        dtype=torch.long,
    )

    for row, example in enumerate(examples):
        inputs[row, : len(example.input_ids)] = torch.tensor(example.input_ids, dtype=torch.long)
        previous_tokens[row, : len(example.previous_ids)] = torch.tensor(
            example.previous_ids,
            dtype=torch.long,
        )
        op_ids = [action.op_id for action in example.actions]
        insert_ids = [
            action.token_id if action.token_id is not None else vocabs.output_pad_id
            for action in example.actions
        ]
        action_target_ops[row, : len(op_ids)] = torch.tensor(op_ids, dtype=torch.long)
        for index, action in enumerate(example.actions):
            if action.op_id == INSERT and action.token_id is not None:
                action_target_insert_tokens[row, index] = action.token_id
        if len(op_ids) > 1:
            action_input_ops[row, 1 : len(op_ids)] = torch.tensor(op_ids[:-1], dtype=torch.long)
            action_input_insert_tokens[row, 1 : len(insert_ids)] = torch.tensor(
                insert_ids[:-1],
                dtype=torch.long,
            )

    return {
        "inputs": inputs,
        "previous_tokens": previous_tokens,
        "action_input_ops": action_input_ops,
        "action_input_insert_tokens": action_input_insert_tokens,
        "action_target_ops": action_target_ops,
        "action_target_insert_tokens": action_target_insert_tokens,
        "input_lengths": input_lengths,
        "previous_lengths": previous_lengths,
        "action_lengths": action_lengths,
    }
