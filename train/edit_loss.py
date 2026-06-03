"""Loss helpers for neural edit transducer training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.edit_data import ACTION_PAD
from train.edit_data import INSERT
from train.edit_data import INSERT_PAD
from train.loss import move_batch_to_device


def compute_edit_loss(
    model,
    batch: dict[str, torch.Tensor],
    insert_loss_weight: float = 1.0,
) -> torch.Tensor:
    op_logits, insert_logits = model(
        batch["inputs"],
        batch["previous_tokens"],
        batch["action_input_ops"],
        batch["action_input_insert_tokens"],
    )
    op_loss = F.cross_entropy(
        op_logits.reshape(-1, op_logits.shape[-1]),
        batch["action_target_ops"].reshape(-1),
        ignore_index=ACTION_PAD,
    )
    insert_targets = batch["action_target_insert_tokens"]
    insert_mask = batch["action_target_ops"] == INSERT
    if not bool(insert_mask.any().item()):
        return op_loss
    insert_loss = F.cross_entropy(
        insert_logits.reshape(-1, insert_logits.shape[-1]),
        insert_targets.reshape(-1),
        ignore_index=INSERT_PAD,
    )
    return op_loss + insert_loss_weight * insert_loss


@torch.no_grad()
def evaluate_average_edit_loss(
    model,
    loader: DataLoader,
    device: torch.device | None = None,
    insert_loss_weight: float = 1.0,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        if device is not None:
            batch = move_batch_to_device(batch, device)
        losses.append(float(compute_edit_loss(model, batch, insert_loss_weight).item()))
    model.train()
    return sum(losses) / len(losses)
