"""Loss helpers for the iterative edit refiner."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.refiner.data import DELETE_PAD
from train.refiner.data import FILL_PAD
from train.refiner.data import INSERT_PAD
from train.common.batch import move_batch_to_device


def compute_refine_loss(
    model,
    batch: dict[str, torch.Tensor],
    insert_loss_weight: float = 1.0,
    fill_loss_weight: float = 1.0,
) -> torch.Tensor:
    outputs = model(
        batch["inputs"],
        batch["hypothesis"],
        placeholder_tokens=batch["placeholders"],
        romaji_pad_mask=batch["input_pad_mask"],
        hypothesis_pad_mask=batch["hypothesis_pad_mask"],
        placeholder_pad_mask=batch["placeholder_pad_mask"],
    )
    delete_logits = outputs["delete_logits"]
    insert_logits = outputs["insert_logits"]
    fill_logits = outputs["fill_logits"]

    delete_loss = F.cross_entropy(
        delete_logits.reshape(-1, delete_logits.shape[-1]),
        batch["delete_target"].reshape(-1),
        ignore_index=DELETE_PAD,
    )
    insert_loss = F.cross_entropy(
        insert_logits.reshape(-1, insert_logits.shape[-1]),
        batch["insert_target"].reshape(-1),
        ignore_index=INSERT_PAD,
    )
    total = delete_loss + insert_loss_weight * insert_loss

    fill_target = batch["fill_target"]
    if bool((fill_target != FILL_PAD).any().item()):
        fill_loss = F.cross_entropy(
            fill_logits.reshape(-1, fill_logits.shape[-1]),
            fill_target.reshape(-1),
            ignore_index=FILL_PAD,
        )
        total = total + fill_loss_weight * fill_loss
    return total


@torch.no_grad()
def evaluate_average_refine_loss(
    model,
    loader: DataLoader,
    device: torch.device | None = None,
    insert_loss_weight: float = 1.0,
    fill_loss_weight: float = 1.0,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        if device is not None:
            batch = move_batch_to_device(batch, device)
        losses.append(
            float(
                compute_refine_loss(
                    model, batch, insert_loss_weight, fill_loss_weight
                ).item()
            )
        )
    model.train()
    return sum(losses) / len(losses)
