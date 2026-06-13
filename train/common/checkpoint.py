"""Checkpoint and artifact helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from dataset.vocab import vocab_from_token_to_id
from train.common.data import TrainingVocabs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def save_vocabs(output_dir: Path, vocabs: TrainingVocabs) -> None:
    write_json(output_dir / "input_vocab.json", vocabs.input_vocab.to_dict())
    write_json(output_dir / "output_vocab.json", vocabs.output_vocab.to_dict())


def has_saved_vocab(vocab_dir) -> bool:
    vocab_dir = Path(vocab_dir)
    return (vocab_dir / "input_vocab.json").exists() and (
        vocab_dir / "output_vocab.json"
    ).exists()


def load_vocabs(vocab_dir) -> TrainingVocabs:
    """Read back the vocab JSONs written by ``save_vocabs`` (special tokens included)."""
    vocab_dir = Path(vocab_dir)
    with (vocab_dir / "input_vocab.json").open("r", encoding="utf-8") as file:
        input_vocab = vocab_from_token_to_id(json.load(file))
    with (vocab_dir / "output_vocab.json").open("r", encoding="utf-8") as file:
        output_vocab = vocab_from_token_to_id(json.load(file))
    return TrainingVocabs(input_vocab=input_vocab, output_vocab=output_vocab)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float | None,
    config,
    scheduler=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "config": asdict(config),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location=None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
