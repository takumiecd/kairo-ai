"""Train Kairo neural edit transducer on JSONL data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from model.edit_transducer import KairoEditTransducer
from train.checkpoint import load_checkpoint
from train.checkpoint import save_checkpoint
from train.checkpoint import save_vocabs
from train.checkpoint import write_json
from train.edit_data import collate_edit_batch
from train.edit_data import load_train_valid_edit_datasets_and_vocabs
from train.edit_loss import compute_edit_loss
from train.edit_loss import evaluate_average_edit_loss
from train.edit_validation import evaluate_edit_decode_cer
from train.loss import move_batch_to_device
from train.train import select_device
from train.train import split_dataset


@dataclass(frozen=True)
class EditTrainConfig:
    data: str
    valid_data: str | None
    output_dir: str
    epochs: int
    batch_size: int
    input_embed_dim: int
    output_embed_dim: int
    action_embed_dim: int
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    limit_examples: int | None
    gradient_clip: float
    insert_loss_weight: float
    valid_decode: str
    valid_cer_samples: int
    valid_cer_every: int
    valid_beam_width: int
    valid_expansion_width: int
    valid_max_actions: int
    edit_penalty: float
    insert_penalty: float
    output_tokenizer: str
    output_vocab_size: int
    output_min_token_frequency: int
    amp: bool
    device: str
    seed: int
    resume: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-embed-dim", type=int, default=64)
    parser.add_argument("--output-embed-dim", type=int, default=64)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--insert-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--valid-decode",
        choices=["none", "greedy", "beam"],
        default="none",
    )
    parser.add_argument("--valid-cer-samples", type=int, default=100)
    parser.add_argument("--valid-cer-every", type=int, default=1)
    parser.add_argument("--valid-beam-width", type=int, default=5)
    parser.add_argument("--valid-expansion-width", type=int, default=5)
    parser.add_argument("--valid-max-actions", type=int, default=128)
    parser.add_argument("--edit-penalty", type=float, default=0.0)
    parser.add_argument("--insert-penalty", type=float, default=0.0)
    parser.add_argument(
        "--output-tokenizer",
        choices=["char", "bpe"],
        default="char",
    )
    parser.add_argument("--output-vocab-size", type=int, default=4000)
    parser.add_argument("--output-min-token-frequency", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def build_edit_loader(dataset, vocabs, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_edit_batch(examples, vocabs),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)

    dataset, explicit_valid_dataset, vocabs = load_train_valid_edit_datasets_and_vocabs(
        args.data,
        args.valid_data,
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
    )
    if args.limit_examples is not None:
        dataset = Subset(dataset, range(min(args.limit_examples, len(dataset))))
        if explicit_valid_dataset is not None:
            explicit_valid_dataset = Subset(
                explicit_valid_dataset,
                range(min(args.limit_examples, len(explicit_valid_dataset))),
            )
    if explicit_valid_dataset is None:
        train_dataset, valid_dataset = split_dataset(
            dataset,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
    else:
        train_dataset = dataset
        valid_dataset = explicit_valid_dataset
    train_loader = build_edit_loader(train_dataset, vocabs, args.batch_size, shuffle=True)
    valid_loader = (
        build_edit_loader(valid_dataset, vocabs, args.batch_size, shuffle=False)
        if len(valid_dataset) > 0
        else None
    )

    config = EditTrainConfig(
        data=str(args.data),
        valid_data=str(args.valid_data) if args.valid_data else None,
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_embed_dim=args.input_embed_dim,
        output_embed_dim=args.output_embed_dim,
        action_embed_dim=args.action_embed_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        limit_examples=args.limit_examples,
        gradient_clip=args.gradient_clip,
        insert_loss_weight=args.insert_loss_weight,
        valid_decode=args.valid_decode,
        valid_cer_samples=args.valid_cer_samples,
        valid_cer_every=args.valid_cer_every,
        valid_beam_width=args.valid_beam_width,
        valid_expansion_width=args.valid_expansion_width,
        valid_max_actions=args.valid_max_actions,
        edit_penalty=args.edit_penalty,
        insert_penalty=args.insert_penalty,
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
        amp=args.amp,
        device=str(device),
        seed=args.seed,
        resume=str(args.resume) if args.resume else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", asdict(config))
    save_vocabs(args.output_dir, vocabs)

    model = KairoEditTransducer(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        input_embed_dim=args.input_embed_dim,
        output_embed_dim=args.output_embed_dim,
        action_embed_dim=args.action_embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    best_valid_loss = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = compute_edit_loss(
                    model,
                    batch,
                    insert_loss_weight=args.insert_loss_weight,
                )
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        train_loss = sum(train_losses) / len(train_losses)
        valid_loss = (
            evaluate_average_edit_loss(
                model,
                valid_loader,
                device=device,
                insert_loss_weight=args.insert_loss_weight,
            )
            if valid_loader is not None
            else train_loss
        )
        valid_cer = None
        if (
            valid_dataset is not None
            and args.valid_decode != "none"
            and args.valid_cer_every > 0
            and epoch % args.valid_cer_every == 0
        ):
            valid_cer = evaluate_edit_decode_cer(
                model,
                valid_dataset,
                vocabs.output_vocab,
                decoder=args.valid_decode,
                max_samples=args.valid_cer_samples,
                beam_width=args.valid_beam_width,
                expansion_width=args.valid_expansion_width,
                max_actions=args.valid_max_actions,
                edit_penalty=args.edit_penalty,
                insert_penalty=args.insert_penalty,
            )
        metrics = [
            f"epoch={epoch}",
            f"train_loss={train_loss:.4f}",
            f"valid_loss={valid_loss:.4f}",
        ]
        if valid_cer is not None:
            metrics.append(f"valid_cer={valid_cer:.4f}")
            metrics.append(f"valid_decode={args.valid_decode}")
            metrics.append(f"valid_cer_samples={args.valid_cer_samples}")
        print(" ".join(metrics))

        checkpoint_path = args.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            train_loss,
            valid_loss,
            config,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                args.output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                train_loss,
                valid_loss,
                config,
            )


if __name__ == "__main__":
    main()
