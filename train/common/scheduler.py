"""共通 LR スケジューラ。

モデル非依存（rnnt / edit / refiner / diffusion すべてが共有）。
``Trainer`` から per-step で step される LambdaLR を組み立てる。warmup は
線形、本体は cosine もしくは inverse-sqrt。``none`` なら ``None`` を返し、
``Trainer`` 側は scheduler を一切触らない（= 既存挙動と同一）。

warmup/本体の境界やステップ総数は学習開始時にしか確定しないので、生成は
``Trainer.fit`` から（``total_steps`` が分かってから）呼ぶ。
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR

SCHEDULER_CHOICES = ("none", "cosine", "inverse_sqrt")


def resolve_warmup_steps(total_steps: int, warmup_steps: int | None, warmup_ratio: float) -> int:
    """明示 ``warmup_steps`` 優先、無ければ ``warmup_ratio * total_steps``。"""
    if warmup_steps is not None:
        return max(0, int(warmup_steps))
    return max(0, int(round(warmup_ratio * total_steps)))


def build_lr_scheduler(
    optimizer,
    *,
    name: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    """per-step で step する LambdaLR を返す（``name == "none"`` なら ``None``）。

    返る乗数は base_lr に対する係数:
      - warmup 区間: 0→1 の線形増加
      - cosine: 1→``min_lr_ratio`` の cosine 減衰
      - inverse_sqrt: warmup ピークから 1/sqrt(step) 減衰（Noam 風）
    """
    if name == "none":
        return None
    if name not in SCHEDULER_CHOICES:
        raise ValueError(f"unknown lr scheduler: {name!r}")

    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        # step は 0 始まり。warmup の最終ステップでちょうど 1.0 になるよう +1。
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if name == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        if name == "inverse_sqrt":
            # warmup ピーク lr を基準に 1/sqrt で減衰。
            peak = max(1, warmup_steps)
            return math.sqrt(peak / float(max(step + 1, peak)))

        return 1.0

    return LambdaLR(optimizer, lr_lambda)
