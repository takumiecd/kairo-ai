"""Loss helpers for discrete text diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.common.batch import move_batch_to_device
from train.diffusion.data import TOKEN_PAD


def compute_diffusion_loss(
    model,
    batch: dict[str, torch.Tensor],
    length_loss_weight: float = 0.25,
) -> torch.Tensor:
    outputs = model(
        batch["inputs"],
        batch["contexts"],
        batch["noisy_canvas"],
        batch["timesteps"],
        input_pad_mask=batch["input_pad_mask"],
        context_pad_mask=batch["context_pad_mask"],
        canvas_pad_mask=batch["canvas_pad_mask"],
    )
    token_logits = outputs["token_logits"]
    token_loss = F.cross_entropy(
        token_logits.reshape(-1, token_logits.shape[-1]),
        batch["token_targets"].reshape(-1),
        ignore_index=TOKEN_PAD,
    )
    length_loss = F.cross_entropy(outputs["length_logits"], batch["length_targets"])
    return token_loss + length_loss_weight * length_loss


@torch.no_grad()
def evaluate_average_diffusion_loss(
    model,
    loader: DataLoader,
    device: torch.device | None = None,
    length_loss_weight: float = 0.25,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        if device is not None:
            batch = move_batch_to_device(batch, device)
        losses.append(float(compute_diffusion_loss(model, batch, length_loss_weight).item()))
    model.train()
    return sum(losses) / len(losses)
