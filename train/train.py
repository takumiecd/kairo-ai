"""Train Kairo RNN-T on generated JSONL data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from model.transducer import KairoTransducer
from train.checkpoint import save_checkpoint
from train.checkpoint import save_vocabs
from train.checkpoint import write_json
from train.checkpoint import load_checkpoint
from train.data import collate_transducer_batch
from train.data import load_train_valid_datasets_and_vocabs
from train.loss import compute_rnnt_loss
from train.loss import evaluate_average_loss
from train.loss import move_batch_to_device
from train.validation import evaluate_decode_cer


@dataclass(frozen=True)
class TrainConfig:
    data: str
    valid_data: str | None
    output_dir: str
    epochs: int
    batch_size: int
    embed_dim: int
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    limit_examples: int | None
    gradient_clip: float
    device: str
    seed: int
    resume: str | None
    valid_decode: str
    valid_cer_samples: int
    valid_cer_every: int
    valid_beam_width: int
    valid_expansion_width: int
    max_len: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--valid-data",
        type=Path,
        default=None,
        help="Optional explicit validation JSONL. Disables internal validation split.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--valid-decode",
        choices=["none", "greedy", "beam"],
        default="none",
        help="Decoder used for validation CER.",
    )
    parser.add_argument("--valid-cer-samples", type=int, default=100)
    parser.add_argument("--valid-cer-every", type=int, default=1)
    parser.add_argument("--valid-beam-width", type=int, default=5)
    parser.add_argument("--valid-expansion-width", type=int, default=5)
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Filter out examples where input or target length exceeds this limit.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def split_dataset(dataset, validation_ratio: float, seed: int):
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0.0, 1.0)")

    indexes = list(range(len(dataset)))
    random.Random(seed).shuffle(indexes)
    valid_size = int(len(indexes) * validation_ratio)
    if validation_ratio > 0.0 and valid_size == 0 and len(indexes) > 1:
        valid_size = 1

    valid_indexes = indexes[:valid_size]
    train_indexes = indexes[valid_size:] or valid_indexes
    return Subset(dataset, train_indexes), Subset(dataset, valid_indexes)


def build_loader(dataset, vocabs, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_transducer_batch(examples, vocabs),
    )


def load_best_valid_loss(output_dir: Path) -> float:
    best_path = output_dir / "checkpoints" / "best.pt"
    if not best_path.exists():
        return float("inf")
    state = load_checkpoint(best_path, map_location="cpu")
    return float(state.get("valid_loss", float("inf")))


def get_resume_model_dims(checkpoint: dict) -> tuple[int, int]:
    config = checkpoint.get("config") or {}
    if "embed_dim" in config and "hidden_dim" in config:
        return int(config["embed_dim"]), int(config["hidden_dim"])

    state_dict = checkpoint["model_state_dict"]
    embed_dim = int(state_dict["encoder_emb.weight"].shape[1])
    hidden_dim = int(state_dict["encoder_lstm.weight_hh_l0"].shape[1])
    return embed_dim, hidden_dim


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    resume_checkpoint = (
        load_checkpoint(args.resume, map_location=device) if args.resume else None
    )
    if resume_checkpoint is not None:
        args.embed_dim, args.hidden_dim = get_resume_model_dims(resume_checkpoint)
        print(
            f"resume_model_dims embed_dim={args.embed_dim} "
            f"hidden_dim={args.hidden_dim}"
        )

    dataset, explicit_valid_dataset, vocabs = load_train_valid_datasets_and_vocabs(
        args.data,
        args.valid_data,
        max_len=args.max_len,
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
    train_loader = build_loader(train_dataset, vocabs, args.batch_size, shuffle=True)
    valid_loader = (
        build_loader(valid_dataset, vocabs, args.batch_size, shuffle=False)
        if len(valid_dataset) > 0
        else None
    )

    config = TrainConfig(
        data=str(args.data),
        valid_data=str(args.valid_data) if args.valid_data else None,
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        limit_examples=args.limit_examples,
        gradient_clip=args.gradient_clip,
        device=str(device),
        seed=args.seed,
        resume=str(args.resume) if args.resume else None,
        valid_decode=args.valid_decode,
        valid_cer_samples=args.valid_cer_samples,
        valid_cer_every=args.valid_cer_every,
        valid_beam_width=args.valid_beam_width,
        valid_expansion_width=args.valid_expansion_width,
        max_len=args.max_len,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", asdict(config))
    save_vocabs(args.output_dir, vocabs)

    model = KairoTransducer(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"resumed_from={args.resume} start_epoch={start_epoch}")

    best_valid_loss = load_best_valid_loss(args.output_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} started. Training on {len(train_loader)} batches...")
        model.train()
        train_losses: list[float] = []
        
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        if has_tqdm:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        else:
            pbar = train_loader

        for i, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_rnnt_loss(model, batch, vocabs.blank_id)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_val = float(loss.item())
            train_losses.append(loss_val)
            if has_tqdm:
                pbar.set_postfix(loss=f"{loss_val:.4f}")
            elif i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}: loss={loss_val:.4f}")

        train_loss = sum(train_losses) / len(train_losses)
        valid_loss = (
            evaluate_average_loss(model, valid_loader, vocabs.blank_id, device)
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
            valid_cer = evaluate_decode_cer(
                model,
                valid_dataset,
                vocabs.output_vocab,
                decoder=args.valid_decode,
                max_samples=args.valid_cer_samples,
                beam_width=args.valid_beam_width,
                expansion_width=args.valid_expansion_width,
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
