"""Dataset and corruption helpers for discrete text diffusion."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.utils.data import Dataset

from dataset.vocab import CharVocab
from train.common.cache import load_or_encode
from train.common.checkpoint import has_saved_vocab
from train.common.checkpoint import load_vocabs
from train.common.data import TrainingVocabs
from train.common.data import load_jsonl_examples
from train.edit.data import build_edit_vocabs_from_records


MASK_TOKEN = "<mask>"
TOKEN_PAD = -100


@dataclass(frozen=True)
class EncodedDiffusionExample:
    input_ids: list[int]
    context_ids: list[int]
    target_ids: list[int]
    input_text: str
    context_text: str
    target_text: str


class JsonlDiffusionDataset(Dataset):
    def __init__(self, examples: list[EncodedDiffusionExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedDiffusionExample:
        return self.examples[index]


def extend_output_vocab(vocab: CharVocab) -> CharVocab:
    id_to_token = list(vocab.id_to_token)
    if MASK_TOKEN not in id_to_token:
        id_to_token.append(MASK_TOKEN)
    return CharVocab(
        token_to_id={token: index for index, token in enumerate(id_to_token)},
        id_to_token=id_to_token,
    )


def mask_id(vocabs: TrainingVocabs) -> int:
    return vocabs.output_vocab.token_to_id[MASK_TOKEN]


def build_diffusion_vocabs_from_records(
    records: list[dict[str, str]],
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
) -> TrainingVocabs:
    vocab_records = [
        {
            "input": record["input"],
            "previous": record.get("context", ""),
            "target": record["target"],
        }
        for record in records
    ]
    base = build_edit_vocabs_from_records(
        vocab_records,
        output_tokenizer=output_tokenizer,
        output_vocab_size=output_vocab_size,
        output_min_token_frequency=output_min_token_frequency,
    )
    return TrainingVocabs(base.input_vocab, extend_output_vocab(base.output_vocab))


def encode_diffusion_records(
    records: list[dict[str, str]],
    vocabs: TrainingVocabs,
    max_positions: int,
    desc: str = "Encoding diffusion records",
) -> JsonlDiffusionDataset:
    examples: list[EncodedDiffusionExample] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc=desc, leave=False)
    except ImportError:
        iterator = records
    for record in iterator:
        context = record.get("context", "")
        example = EncodedDiffusionExample(
            input_ids=vocabs.input_vocab.encode(record["input"]),
            context_ids=vocabs.output_vocab.encode(context),
            target_ids=vocabs.output_vocab.encode(record["target"]),
            input_text=record["input"],
            context_text=context,
            target_text=record["target"],
        )
        if (
            0 < len(example.input_ids) <= max_positions
            and 0 < len(example.target_ids) <= max_positions
            and len(example.context_ids) <= max_positions
        ):
            examples.append(example)
    return JsonlDiffusionDataset(examples)


def load_train_valid_diffusion_datasets_and_vocabs(
    train_path,
    valid_path=None,
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
    max_positions: int = 256,
    vocab_dir=None,
    cache_dir=None,
) -> tuple[JsonlDiffusionDataset, JsonlDiffusionDataset | None, TrainingVocabs]:
    print(f"Loading train records from {train_path}...", flush=True)
    train_records = load_jsonl_examples(train_path)
    print(f"Loaded {len(train_records)} train records", flush=True)
    if valid_path is not None:
        print(f"Loading valid records from {valid_path}...", flush=True)
        valid_records = load_jsonl_examples(valid_path)
        print(f"Loaded {len(valid_records)} valid records", flush=True)
    else:
        valid_records = []
    if vocab_dir is not None and has_saved_vocab(vocab_dir):
        print(f"Reusing vocab from {vocab_dir}", flush=True)
        vocabs = load_vocabs(vocab_dir)
    else:
        print(
            f"Building vocab (tokenizer={output_tokenizer}, size={output_vocab_size})...",
            flush=True,
        )
        vocabs = build_diffusion_vocabs_from_records(
            train_records + valid_records,
            output_tokenizer=output_tokenizer,
            output_vocab_size=output_vocab_size,
            output_min_token_frequency=output_min_token_frequency,
        )
    print(
        f"Built input vocab={len(vocabs.input_vocab.id_to_token)} "
        f"output vocab={len(vocabs.output_vocab.id_to_token)}",
        flush=True,
    )
    train = load_or_encode(
        train_path,
        vocabs,
        cache_dir,
        "train",
        encode_fn=lambda: encode_diffusion_records(
            train_records, vocabs, max_positions, desc="Encoding train"
        ),
        rebuild_fn=JsonlDiffusionDataset,
        extra_key=f"P{max_positions}",
    )
    valid = (
        load_or_encode(
            valid_path,
            vocabs,
            cache_dir,
            "valid",
            encode_fn=lambda: encode_diffusion_records(
                valid_records, vocabs, max_positions, desc="Encoding valid"
            ),
            rebuild_fn=JsonlDiffusionDataset,
            extra_key=f"P{max_positions}",
        )
        if valid_path is not None
        else None
    )
    print(
        f"Encoded train={len(train)}/{len(train_records)}"
        + (
            f" valid={len(valid)}/{len(valid_records)}"
            if valid is not None
            else ""
        ),
        flush=True,
    )
    return train, valid, vocabs


def corrupt_tokens(
    target_ids: list[int],
    timestep: int,
    diffusion_steps: int,
    mask_token_id: int,
    random_token_ids: list[int],
    rng: random.Random,
) -> list[int]:
    probability = timestep / diffusion_steps
    noisy = list(target_ids)
    changed = False
    for index in range(len(noisy)):
        if rng.random() >= probability:
            continue
        changed = True
        if rng.random() < 0.9:
            noisy[index] = mask_token_id
        else:
            noisy[index] = rng.choice(random_token_ids)
    if not changed and noisy:
        noisy[rng.randrange(len(noisy))] = mask_token_id
    return noisy


def collate_diffusion_batch(
    examples: list[EncodedDiffusionExample],
    vocabs: TrainingVocabs,
    diffusion_steps: int,
    rng: random.Random | None = None,
) -> dict[str, torch.Tensor]:
    rng = rng or random
    batch_size = len(examples)
    max_input = max(len(example.input_ids) for example in examples)
    max_context = max((len(example.context_ids) for example in examples), default=0)
    max_target = max(len(example.target_ids) for example in examples)
    random_ids = [
        index
        for index, token in enumerate(vocabs.output_vocab.id_to_token)
        if not token.startswith("<")
    ]

    inputs = torch.full((batch_size, max_input), vocabs.input_pad_id, dtype=torch.long)
    contexts = torch.full((batch_size, max_context), vocabs.output_pad_id, dtype=torch.long)
    noisy_canvas = torch.full(
        (batch_size, max_target), vocabs.output_pad_id, dtype=torch.long
    )
    token_targets = torch.full((batch_size, max_target), TOKEN_PAD, dtype=torch.long)
    input_pad_mask = torch.ones((batch_size, max_input), dtype=torch.bool)
    context_pad_mask = torch.ones((batch_size, max_context), dtype=torch.bool)
    canvas_pad_mask = torch.ones((batch_size, max_target), dtype=torch.bool)
    timesteps = torch.empty(batch_size, dtype=torch.long)
    length_targets = torch.empty(batch_size, dtype=torch.long)

    for row, example in enumerate(examples):
        timestep = rng.randint(1, diffusion_steps)
        noisy = corrupt_tokens(
            example.target_ids,
            timestep,
            diffusion_steps,
            mask_id(vocabs),
            random_ids,
            rng,
        )
        n_input = len(example.input_ids)
        n_context = len(example.context_ids)
        n_target = len(example.target_ids)
        inputs[row, :n_input] = torch.tensor(example.input_ids)
        if n_context:
            contexts[row, :n_context] = torch.tensor(example.context_ids)
        noisy_canvas[row, :n_target] = torch.tensor(noisy)
        token_targets[row, :n_target] = torch.tensor(example.target_ids)
        input_pad_mask[row, :n_input] = False
        context_pad_mask[row, :n_context] = False
        canvas_pad_mask[row, :n_target] = False
        timesteps[row] = timestep
        length_targets[row] = n_target

    return {
        "inputs": inputs,
        "contexts": contexts,
        "noisy_canvas": noisy_canvas,
        "token_targets": token_targets,
        "timesteps": timesteps,
        "length_targets": length_targets,
        "input_pad_mask": input_pad_mask,
        "context_pad_mask": context_pad_mask,
        "canvas_pad_mask": canvas_pad_mask,
    }
