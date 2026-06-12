"""RNN-T loss and evaluation helpers."""

from __future__ import annotations

import torch
import torchaudio.functional as F
from torch.utils.data import DataLoader

from train.common.batch import move_batch_to_device


def compute_rnnt_loss(model, batch: dict[str, torch.Tensor], blank_id: int) -> torch.Tensor:
    logits = model(batch["inputs"], batch["prediction_inputs"])
    return F.rnnt_loss(
        logits=logits.float(),
        targets=batch["targets"],
        logit_lengths=batch["input_lengths"],
        target_lengths=batch["target_lengths"],
        blank=blank_id,
        reduction="mean",
        fused_log_softmax=True,
    )


@torch.no_grad()
def evaluate_average_loss(
    model,
    loader: DataLoader,
    blank_id: int,
    device: torch.device | None = None,
    amp: bool = False,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        if device is not None:
            batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(
            "cuda",
            enabled=amp and device is not None and device.type == "cuda",
        ):
            loss = compute_rnnt_loss(model, batch, blank_id)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)
