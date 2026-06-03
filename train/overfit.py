"""Small RNN-T smoke training loop for generated JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio.functional as F
from torch.utils.data import DataLoader

from model.transducer import KairoTransducer
from train.data import collate_transducer_batch
from train.data import load_dataset_and_vocabs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset, vocabs = load_dataset_and_vocabs(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_transducer_batch(examples, vocabs),
    )
    iterator = iter(loader)

    model = KairoTransducer(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    first_loss: float | None = None
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["inputs"], batch["prediction_inputs"])
        loss = F.rnnt_loss(
            logits=logits,
            targets=batch["targets"],
            logit_lengths=batch["input_lengths"],
            target_lengths=batch["target_lengths"],
            blank=vocabs.blank_id,
            reduction="mean",
            fused_log_softmax=True,
        )
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss
        print(f"step={step} loss={last_loss:.4f}")

    print(f"first_loss={first_loss:.4f} last_loss={last_loss:.4f}")


if __name__ == "__main__":
    main()
