"""Train Kairo neural edit transducer on JSONL data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import random

import torch
from torch.utils.data import Subset

from model.edit_transducer import KairoEditTransducer
from train.common.checkpoint import load_checkpoint
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.edit.data import collate_edit_batch
from train.edit.data import load_train_valid_edit_datasets_and_vocabs
from train.edit.loss import compute_edit_loss
from train.edit.loss import evaluate_average_edit_loss
from train.edit.validation import evaluate_edit_decode_cer
from train.common.engine import Trainer
from train.common.engine import add_common_args
from train.common.engine import build_loader
from train.common.engine import restore_training_state
from train.common.engine import select_device
from train.common.engine import split_dataset


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
    add_common_args(parser)
    # edit transducer 固有のフラグ。
    parser.add_argument("--input-embed-dim", type=int, default=64)
    parser.add_argument("--output-embed-dim", type=int, default=64)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--insert-loss-weight", type=float, default=1.0)
    parser.add_argument("--valid-max-actions", type=int, default=128)
    parser.add_argument("--edit-penalty", type=float, default=0.0)
    parser.add_argument("--insert-penalty", type=float, default=0.0)
    return parser.parse_args()


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
            dataset, validation_ratio=args.validation_ratio, seed=args.seed
        )
    else:
        train_dataset = dataset
        valid_dataset = explicit_valid_dataset

    collate = lambda examples: collate_edit_batch(examples, vocabs)
    train_loader = build_loader(train_dataset, collate, args.batch_size, shuffle=True)
    valid_loader = (
        build_loader(valid_dataset, collate, args.batch_size, shuffle=False)
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
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        start_epoch = restore_training_state(model, optimizer, checkpoint)

    def cer_fn(epoch: int) -> float | None:
        if (
            valid_dataset is None
            or args.valid_decode == "none"
            or args.valid_cer_every <= 0
            or epoch % args.valid_cer_every != 0
        ):
            return None
        return evaluate_edit_decode_cer(
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
    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        loss_fn=lambda m, batch: compute_edit_loss(
            m, batch, insert_loss_weight=args.insert_loss_weight
        ),
        start_epoch=start_epoch,
        valid_loss_fn=(
            (
                lambda _epoch: evaluate_average_edit_loss(
                    model, valid_loader, device=device, insert_loss_weight=args.insert_loss_weight
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
