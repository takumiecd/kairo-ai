"""LoRA (Low-Rank Adaptation) for the Kairo RNN-T.

Personalization fine-tunes a *frozen* base model with small low-rank adapters
trained on confirmed user feedback (see ``dataset/source_feedback.py`` and the
kairo-ai feedback schema). Only the adapter weights are trained and exported, so
a personal adapter is tiny next to the ~7M-parameter base.

We target the **joint network** linears (``joint_fc1`` / ``joint_fc2``) by
default: they are invoked directly in :meth:`KairoTransducer.forward`, so the
adapter reliably shapes the output distribution. The Transformer attention
projections are *not* targeted -- ``torch.nn.functional.multi_head_attention_forward``
reads their weights directly rather than calling the submodule, so wrapping them
would be a no-op.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("joint_fc1", "joint_fc2")


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update.

    ``y = base(x) + (dropout(x) @ A^T @ B^T) * (alpha / r)``. ``B`` is zero-init,
    so the adapter is a no-op until trained (the base output is reproduced exactly).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_a = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # lora_b stays zero -> zero initial delta.
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return base_out + delta * self.scaling


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> list[str]:
    """Replace target ``nn.Linear`` submodules in-place with :class:`LoRALinear`.

    A linear is targeted when any string in ``targets`` is a substring of its
    fully-qualified name. Returns the list of replaced module names.
    """
    to_replace: list[tuple[nn.Module, str, str]] = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if any(target in full_name for target in targets):
                to_replace.append((module, child_name, full_name))

    replaced: list[str] = []
    for parent, child_name, full_name in to_replace:
        base = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        replaced.append(full_name)
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter except the LoRA adapters."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter tensors -- what gets exported as the personal adapter."""
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_" in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load adapter tensors into a model that already has LoRA injected."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if "lora_" in key]
    if unexpected:
        raise ValueError(f"unexpected LoRA keys in adapter: {unexpected}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
