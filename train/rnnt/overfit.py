"""Small RNN-T smoke training loop for generated JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from model.transducer import KairoTransducer
from train.rnnt.data import collate_transducer_batch
from train.rnnt.data import load_dataset_and_vocabs
from train.rnnt.loss import compute_rnnt_loss
from train.rnnt.loss import evaluate_average_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--prediction-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--prediction-layers", type=int, default=1)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Use only the first N examples for tiny overfit checks.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate average loss every N steps. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset, vocabs = load_dataset_and_vocabs(args.data)
    if args.max_examples is not None:
        dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))

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
        encoder_type=args.encoder_type,
        prediction_type=args.prediction_type,
        encoder_layers=args.encoder_layers,
        prediction_layers=args.prediction_layers,
        input_pad_id=vocabs.input_vocab.token_to_id["<pad>"],
        output_pad_id=vocabs.output_vocab.token_to_id["<pad>"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_transducer_batch(examples, vocabs),
    )
    initial_eval_loss = evaluate_average_loss(model, eval_loader, vocabs.blank_id)
    print(f"initial_eval_loss={initial_eval_loss:.4f}")

    last_loss = 0.0
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_rnnt_loss(model, batch, vocabs.blank_id)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        print(f"step={step} loss={last_loss:.4f}")
        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_loss = evaluate_average_loss(model, eval_loader, vocabs.blank_id)
            print(f"step={step} eval_loss={eval_loss:.4f}")

    final_eval_loss = evaluate_average_loss(model, eval_loader, vocabs.blank_id)
    print(
        f"initial_eval_loss={initial_eval_loss:.4f} "
        f"final_eval_loss={final_eval_loss:.4f} "
        f"last_train_loss={last_loss:.4f}"
    )


if __name__ == "__main__":
    main()
