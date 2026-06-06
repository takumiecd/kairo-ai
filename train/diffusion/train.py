"""Train the context-aware discrete diffusion IME model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Subset

from decode.diffusion import diffusion_decode
from eval.metrics import mean_cer
from model.diffusion import KairoDiffusionModel
from train.common.checkpoint import load_checkpoint
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.common.engine import Trainer
from train.common.engine import add_common_args
from train.common.engine import build_loader
from train.common.engine import restore_training_state
from train.common.engine import select_device
from train.common.engine import split_dataset
from train.common.validation import unwrap_subset
from train.diffusion.data import collate_diffusion_batch
from train.diffusion.data import load_train_valid_diffusion_datasets_and_vocabs
from train.diffusion.loss import compute_diffusion_loss
from train.diffusion.loss import evaluate_average_diffusion_loss


@dataclass(frozen=True)
class DiffusionTrainConfig:
    data: str
    valid_data: str | None
    output_dir: str
    epochs: int
    batch_size: int
    model_dim: int
    input_embed_dim: int
    output_embed_dim: int
    num_heads: int
    num_input_layers: int
    num_context_layers: int
    num_canvas_layers: int
    feedforward_dim: int
    dropout: float
    max_positions: int
    diffusion_steps: int
    length_loss_weight: float
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    limit_examples: int | None
    gradient_clip: float
    valid_decode: str
    valid_cer_samples: int
    valid_cer_every: int
    output_tokenizer: str
    output_vocab_size: int
    output_min_token_frequency: int
    amp: bool
    device: str
    seed: int
    resume: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--input-embed-dim", type=int, default=64)
    parser.add_argument("--output-embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-input-layers", type=int, default=3)
    parser.add_argument("--num-context-layers", type=int, default=2)
    parser.add_argument("--num-canvas-layers", type=int, default=3)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--length-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Reuse a saved vocab dir (input_vocab.json/output_vocab.json) instead of rebuilding.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to cache encoded datasets. Defaults to <data dir>/.kairo_cache.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the encoded-dataset cache (always re-encode).",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_diffusion_cer(model, dataset, vocabs, max_samples: int) -> float:
    base_dataset, indexes = unwrap_subset(dataset)
    selected = indexes[:max_samples]
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    try:
        from tqdm import tqdm
        iterable = tqdm(selected, desc="Validation CER decode", leave=False)
    except ImportError:
        iterable = selected
    for index in iterable:
        example = base_dataset[index]
        predictions.append(
            diffusion_decode(
                model,
                example.input_ids,
                example.context_ids,
                vocabs.output_vocab,
            )
        )
        references.append(example.target_text)
    model.train()
    return mean_cer(predictions, references)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    print(f"Using device={device}", flush=True)

    # 語彙の再利用: --vocab 明示、または resume 時はその run の output_dir を流用
    # （再構築せず checkpoint と同じ語彙を保証＆BPE 構築を省略）。
    vocab_dir = args.vocab
    if vocab_dir is None and args.resume is not None:
        vocab_dir = args.output_dir
    cache_dir = (
        None
        if args.no_cache
        else (args.cache_dir or Path(args.data).resolve().parent / ".kairo_cache")
    )

    print("Preparing diffusion datasets...", flush=True)
    dataset, explicit_valid_dataset, vocabs = load_train_valid_diffusion_datasets_and_vocabs(
        args.data,
        args.valid_data,
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
        max_positions=args.max_positions,
        vocab_dir=vocab_dir,
        cache_dir=cache_dir,
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
            dataset, args.validation_ratio, args.seed
        )
    else:
        train_dataset, valid_dataset = dataset, explicit_valid_dataset

    collate = lambda examples: collate_diffusion_batch(
        examples, vocabs, args.diffusion_steps
    )
    train_loader = build_loader(train_dataset, collate, args.batch_size, shuffle=True)
    valid_loader = (
        build_loader(valid_dataset, collate, args.batch_size, shuffle=False)
        if len(valid_dataset) > 0
        else None
    )
    print(
        f"Prepared train={len(train_dataset)} valid={len(valid_dataset)} "
        f"train_batches={len(train_loader)} "
        f"valid_batches={len(valid_loader) if valid_loader is not None else 0}",
        flush=True,
    )

    config = DiffusionTrainConfig(
        data=str(args.data),
        valid_data=str(args.valid_data) if args.valid_data else None,
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        input_embed_dim=args.input_embed_dim,
        output_embed_dim=args.output_embed_dim,
        num_heads=args.num_heads,
        num_input_layers=args.num_input_layers,
        num_context_layers=args.num_context_layers,
        num_canvas_layers=args.num_canvas_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_positions=args.max_positions,
        diffusion_steps=args.diffusion_steps,
        length_loss_weight=args.length_loss_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        limit_examples=args.limit_examples,
        gradient_clip=args.gradient_clip,
        valid_decode=args.valid_decode,
        valid_cer_samples=args.valid_cer_samples,
        valid_cer_every=args.valid_cer_every,
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

    print("Building diffusion model...", flush=True)
    model = KairoDiffusionModel(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        model_dim=args.model_dim,
        input_embed_dim=args.input_embed_dim,
        output_embed_dim=args.output_embed_dim,
        num_heads=args.num_heads,
        num_input_layers=args.num_input_layers,
        num_context_layers=args.num_context_layers,
        num_canvas_layers=args.num_canvas_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_positions=args.max_positions,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Built diffusion model parameters={parameter_count:,}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        start_epoch = restore_training_state(model, optimizer, checkpoint)

    def cer_fn(epoch: int) -> float | None:
        if (
            valid_loader is None
            or args.valid_decode == "none"
            or args.valid_cer_every <= 0
            or epoch % args.valid_cer_every != 0
        ):
            return None
        return evaluate_diffusion_cer(
            model, valid_dataset, vocabs, args.valid_cer_samples
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        config=config,
        amp=args.amp,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        resume_scheduler_state=(
            checkpoint.get("scheduler_state_dict") if args.resume else None
        ),
    )
    print(
        f"Starting training epochs={args.epochs} batch_size={args.batch_size} "
        f"diffusion_steps={args.diffusion_steps} amp={trainer.amp_enabled}",
        flush=True,
    )
    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        loss_fn=lambda m, batch: compute_diffusion_loss(
            m, batch, args.length_loss_weight
        ),
        valid_loss_fn=(
            (
                lambda: evaluate_average_diffusion_loss(
                    model, valid_loader, device, args.length_loss_weight
                )
            )
            if valid_loader is not None
            else None
        ),
        cer_fn=cer_fn,
        gradient_clip=args.gradient_clip,
    )


if __name__ == "__main__":
    main()
