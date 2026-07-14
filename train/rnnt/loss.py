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


def compute_rnnt_loss_with_profile(
    model, batch: dict[str, torch.Tensor], blank_id: int
) -> torch.Tensor:
    """段階B(プロファイル条件付け)版の :func:`compute_rnnt_loss`。

    ``batch`` は ``train.rnnt.profile_data.collate_profile_transducer_batch``
    が作るフラットな ``profile_*`` キーを持つ想定。``model`` は
    ``profile_conditioning=True`` で構築された :class:`KairoTransducer`。
    """
    # ローカル import: train/rnnt/loss.py はプロファイル非対応の学習からも
    # 使われるため、循環 import を避けてここでだけ依存する。
    from train.rnnt.profile_data import extract_profile_features

    profile_features = extract_profile_features(batch)
    logits = model(batch["inputs"], batch["prediction_inputs"], profile_features=profile_features)
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
    loss_fn=compute_rnnt_loss,
) -> float:
    """``loss_fn`` は既定で :func:`compute_rnnt_loss`。段階B学習からは
    :func:`compute_rnnt_loss_with_profile` を渡して再利用する。
    """
    model.eval()
    losses: list[float] = []
    for batch in loader:
        if device is not None:
            batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(
            "cuda",
            enabled=amp and device is not None and device.type == "cuda",
        ):
            loss = loss_fn(model, batch, blank_id)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)
