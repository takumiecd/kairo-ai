"""Checkpoint and artifact helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from train.common.data import TrainingVocabs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def save_vocabs(output_dir: Path, vocabs: TrainingVocabs) -> None:
    write_json(output_dir / "input_vocab.json", vocabs.input_vocab.to_dict())
    write_json(output_dir / "output_vocab.json", vocabs.output_vocab.to_dict())


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(path: Path, map_location=None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
