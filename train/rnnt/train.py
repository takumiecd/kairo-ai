"""Train Kairo RNN-T on generated JSONL data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import random

import torch
from torch.utils.data import Subset

from model.transducer import KairoTransducer
from train.common.checkpoint import load_checkpoint
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.rnnt.data import collate_transducer_batch
from train.rnnt.data import load_train_valid_datasets_and_vocabs
from train.common.engine import Trainer
from train.common.engine import add_common_args
from train.common.engine import build_loader
from train.common.engine import restore_training_state
from train.common.engine import select_device
from train.common.engine import split_dataset
from train.rnnt.loss import compute_rnnt_loss
from train.rnnt.loss import evaluate_average_loss
from train.rnnt.validation import evaluate_decode_cer

# 後方互換: テストが train.train から import している共通シンボル。
from train.common.engine import load_best_valid_loss  # noqa: F401


@dataclass(frozen=True)
class TrainConfig:
    data: str
    valid_data: str | None
    output_dir: str
    epochs: int
    batch_size: int
    embed_dim: int
    hidden_dim: int
    input_embed_dim: int
    output_embed_dim: int
    encoder_hidden_dim: int
    prediction_hidden_dim: int
    joint_hidden_dim: int
    encoder_type: str
    prediction_type: str
    encoder_layers: int
    prediction_layers: int
    num_heads: int
    feedforward_dim: int | None
    dropout: float
    max_positions: int
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
    output_tokenizer: str
    output_vocab_size: int
    output_min_token_frequency: int
    max_len: int | None
    amp: bool


@dataclass(frozen=True)
class ModelDims:
    input_embed_dim: int
    output_embed_dim: int
    encoder_hidden_dim: int
    prediction_hidden_dim: int
    joint_hidden_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    # RNN-T 固有の次元フラグ。
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--input-embed-dim", type=int, default=None)
    parser.add_argument("--output-embed-dim", type=int, default=None)
    parser.add_argument("--encoder-hidden-dim", type=int, default=None)
    parser.add_argument("--prediction-hidden-dim", type=int, default=None)
    parser.add_argument("--joint-hidden-dim", type=int, default=None)
    # Encoder / Prediction の中身を切り替える（docs/MODEL_DESIGN.md 参照）。
    parser.add_argument("--encoder-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--prediction-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--prediction-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--feedforward-dim", type=int, default=None, help="Transformer FFN dim (default 4x model dim).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--max-positions", type=int, default=256, help="Transformer max sequence length.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Filter out examples where input or target length exceeds this limit.",
    )
    return parser.parse_args()


def resolve_model_dims(args: argparse.Namespace) -> ModelDims:
    return ModelDims(
        input_embed_dim=args.input_embed_dim or args.embed_dim,
        output_embed_dim=args.output_embed_dim or args.embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim or args.hidden_dim,
        prediction_hidden_dim=args.prediction_hidden_dim or args.hidden_dim,
        joint_hidden_dim=args.joint_hidden_dim or args.hidden_dim,
    )


def get_resume_model_dims(checkpoint: dict) -> ModelDims:
    config = checkpoint.get("config") or {}
    if all(
        key in config
        for key in (
            "input_embed_dim",
            "output_embed_dim",
            "encoder_hidden_dim",
            "prediction_hidden_dim",
            "joint_hidden_dim",
        )
    ):
        return ModelDims(
            input_embed_dim=int(config["input_embed_dim"]),
            output_embed_dim=int(config["output_embed_dim"]),
            encoder_hidden_dim=int(config["encoder_hidden_dim"]),
            prediction_hidden_dim=int(config["prediction_hidden_dim"]),
            joint_hidden_dim=int(config["joint_hidden_dim"]),
        )
    if "embed_dim" in config and "hidden_dim" in config:
        embed_dim = int(config["embed_dim"])
        hidden_dim = int(config["hidden_dim"])
        return ModelDims(
            input_embed_dim=embed_dim,
            output_embed_dim=embed_dim,
            encoder_hidden_dim=hidden_dim,
            prediction_hidden_dim=hidden_dim,
            joint_hidden_dim=hidden_dim,
        )

    state_dict = checkpoint["model_state_dict"]
    return ModelDims(
        input_embed_dim=int(state_dict["encoder_emb.weight"].shape[1]),
        output_embed_dim=int(state_dict["pred_emb.weight"].shape[1]),
        encoder_hidden_dim=int(state_dict["encoder_lstm.weight_hh_l0"].shape[1]),
        prediction_hidden_dim=int(state_dict["pred_lstm.weight_hh_l0"].shape[1]),
        joint_hidden_dim=int(state_dict["joint_fc1.weight"].shape[0]),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)

    resume_checkpoint = (
        load_checkpoint(args.resume, map_location=device) if args.resume else None
    )
    if resume_checkpoint is not None:
        resume_config = resume_checkpoint.get("config") or {}
        resume_dims = get_resume_model_dims(resume_checkpoint)
        args.input_embed_dim = resume_dims.input_embed_dim
        args.output_embed_dim = resume_dims.output_embed_dim
        args.encoder_hidden_dim = resume_dims.encoder_hidden_dim
        args.prediction_hidden_dim = resume_dims.prediction_hidden_dim
        args.joint_hidden_dim = resume_dims.joint_hidden_dim
        args.encoder_type = resume_config.get("encoder_type", args.encoder_type)
        args.prediction_type = resume_config.get("prediction_type", args.prediction_type)
        args.encoder_layers = int(resume_config.get("encoder_layers", args.encoder_layers))
        args.prediction_layers = int(resume_config.get("prediction_layers", args.prediction_layers))
        args.num_heads = int(resume_config.get("num_heads", args.num_heads))
        args.feedforward_dim = resume_config.get("feedforward_dim", args.feedforward_dim)
        args.dropout = float(resume_config.get("dropout", args.dropout))
        args.max_positions = int(resume_config.get("max_positions", args.max_positions))
        args.output_tokenizer = resume_config.get("output_tokenizer", args.output_tokenizer)
        args.output_vocab_size = int(
            resume_config.get("output_vocab_size", args.output_vocab_size)
        )
        args.output_min_token_frequency = int(
            resume_config.get("output_min_token_frequency", args.output_min_token_frequency)
        )
        print(
            f"resume_model_dims input_embed_dim={args.input_embed_dim} "
            f"output_embed_dim={args.output_embed_dim} "
            f"encoder_hidden_dim={args.encoder_hidden_dim} "
            f"prediction_hidden_dim={args.prediction_hidden_dim} "
            f"joint_hidden_dim={args.joint_hidden_dim}"
        )
    dims = resolve_model_dims(args)

    dataset, explicit_valid_dataset, vocabs = load_train_valid_datasets_and_vocabs(
        args.data,
        args.valid_data,
        max_len=args.max_len,
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

    collate = lambda examples: collate_transducer_batch(examples, vocabs)
    train_loader = build_loader(train_dataset, collate, args.batch_size, shuffle=True)
    valid_loader = (
        build_loader(valid_dataset, collate, args.batch_size, shuffle=False)
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
        input_embed_dim=dims.input_embed_dim,
        output_embed_dim=dims.output_embed_dim,
        encoder_hidden_dim=dims.encoder_hidden_dim,
        prediction_hidden_dim=dims.prediction_hidden_dim,
        joint_hidden_dim=dims.joint_hidden_dim,
        encoder_type=args.encoder_type,
        prediction_type=args.prediction_type,
        encoder_layers=args.encoder_layers,
        prediction_layers=args.prediction_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_positions=args.max_positions,
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
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
        max_len=args.max_len,
        amp=args.amp,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", asdict(config))
    save_vocabs(args.output_dir, vocabs)

    model = KairoTransducer(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        input_embed_dim=dims.input_embed_dim,
        output_embed_dim=dims.output_embed_dim,
        encoder_hidden_dim=dims.encoder_hidden_dim,
        prediction_hidden_dim=dims.prediction_hidden_dim,
        joint_hidden_dim=dims.joint_hidden_dim,
        encoder_type=args.encoder_type,
        prediction_type=args.prediction_type,
        encoder_layers=args.encoder_layers,
        prediction_layers=args.prediction_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_positions=args.max_positions,
        input_pad_id=vocabs.input_vocab.token_to_id["<pad>"],
        output_pad_id=vocabs.output_vocab.token_to_id["<pad>"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_epoch = 1
    if resume_checkpoint is not None:
        start_epoch = restore_training_state(model, optimizer, resume_checkpoint)
        print(f"resumed_from={args.resume} start_epoch={start_epoch}")

    def cer_fn(epoch: int) -> float | None:
        if (
            valid_loader is None
            or args.valid_decode == "none"
            or args.valid_cer_every <= 0
            or epoch % args.valid_cer_every != 0
        ):
            return None
        return evaluate_decode_cer(
            model,
            valid_dataset,
            vocabs.output_vocab,
            decoder=args.valid_decode,
            max_samples=args.valid_cer_samples,
            beam_width=args.valid_beam_width,
            expansion_width=args.valid_expansion_width,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        config=config,
        amp=args.amp,
    )
    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        loss_fn=lambda m, batch: compute_rnnt_loss(m, batch, vocabs.blank_id),
        start_epoch=start_epoch,
        valid_loss_fn=(
            (lambda: evaluate_average_loss(model, valid_loader, vocabs.blank_id, device))
            if valid_loader is not None
            else None
        ),
        cer_fn=cer_fn,
        gradient_clip=args.gradient_clip,
    )


if __name__ == "__main__":
    main()
