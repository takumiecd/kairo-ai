"""段階B(プロファイル条件付け)版の RNN-T 学習エントリポイント。

``train/rnnt/train.py`` を壊さない別エントリポイントとして用意する
(docs/PROFILE.md §4, §5)。``dataset/profile_stream.py`` が生成した
プロファイル付き jsonl を読み、``KairoTransducer(profile_conditioning=True)``
を ``train.common.engine.Trainer`` で学習する。

使い方::

    python -m train.rnnt.profile \\
        --data data/profile_stream.jsonl \\
        --output-dir out/profile_run --epochs 5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import random

import torch

from model.profile_encoder import DEFAULT_TOP_K
from model.transducer import KairoTransducer
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.common.engine import Trainer
from train.common.engine import add_common_args
from train.common.engine import build_loader
from train.common.engine import select_device
from train.common.engine import split_dataset
from train.rnnt.loss import compute_rnnt_loss_with_profile
from train.rnnt.loss import evaluate_average_loss
from train.rnnt.profile_data import DEFAULT_PROFILE_DROP_RATE
from train.rnnt.profile_data import collate_profile_transducer_batch
from train.rnnt.profile_data import load_profile_dataset_and_vocabs


@dataclass(frozen=True)
class ProfileTrainConfig:
    data: str
    output_dir: str
    epochs: int
    batch_size: int
    embed_dim: int
    hidden_dim: int
    encoder_type: str
    prediction_type: str
    encoder_layers: int
    prediction_layers: int
    profile_top_k: int
    profile_half_life: int
    profile_drop_rate: float
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    gradient_clip: float
    device: str
    seed: int
    output_tokenizer: str
    output_vocab_size: int
    output_min_token_frequency: int
    amp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--prediction-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--prediction-layers", type=int, default=1)
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="e(u) に使う top-K 語数(PROFILE.md §4)。",
    )
    parser.add_argument(
        "--profile-half-life",
        type=int,
        default=100_000,
        help="top-K 語選定に使う lazy decay の半減期(総確定文字数単位)。",
    )
    parser.add_argument(
        "--profile-drop-rate",
        type=float,
        default=DEFAULT_PROFILE_DROP_RATE,
        help="確率 p_drop で e(u) を e(u_0) に置換する頑健化(PROFILE.md §5)。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)

    dataset, vocabs = load_profile_dataset_and_vocabs(
        args.data,
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
    )
    if args.limit_examples is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.limit_examples, len(dataset))))

    train_dataset, valid_dataset = split_dataset(
        dataset, validation_ratio=args.validation_ratio, seed=args.seed
    )

    train_rng = random.Random(args.seed)
    train_collate = lambda examples: collate_profile_transducer_batch(
        examples,
        vocabs,
        top_k=args.profile_top_k,
        half_life=args.profile_half_life,
        profile_drop_rate=args.profile_drop_rate,
        rng=train_rng,
    )
    # 検証はドロップなし(u_0 置換をせず実際のプロファイルで評価する)。
    valid_collate = lambda examples: collate_profile_transducer_batch(
        examples,
        vocabs,
        top_k=args.profile_top_k,
        half_life=args.profile_half_life,
        profile_drop_rate=0.0,
    )

    train_loader = build_loader(train_dataset, train_collate, args.batch_size, shuffle=True)
    valid_loader = (
        build_loader(valid_dataset, valid_collate, args.batch_size, shuffle=False)
        if len(valid_dataset) > 0
        else None
    )

    config = ProfileTrainConfig(
        data=str(args.data),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        prediction_type=args.prediction_type,
        encoder_layers=args.encoder_layers,
        prediction_layers=args.prediction_layers,
        profile_top_k=args.profile_top_k,
        profile_half_life=args.profile_half_life,
        profile_drop_rate=args.profile_drop_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        gradient_clip=args.gradient_clip,
        device=str(device),
        seed=args.seed,
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
        amp=args.amp,
    )

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
        profile_conditioning=True,
        profile_top_k=args.profile_top_k,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", asdict(config))
    save_vocabs(args.output_dir, vocabs)

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
    )
    trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        loss_fn=lambda m, batch: compute_rnnt_loss_with_profile(m, batch, vocabs.blank_id),
        valid_loss_fn=(
            (
                lambda epoch: evaluate_average_loss(
                    model,
                    valid_loader,
                    vocabs.blank_id,
                    device,
                    amp=args.amp,
                    loss_fn=compute_rnnt_loss_with_profile,
                )
            )
            if valid_loader is not None
            else None
        ),
        gradient_clip=args.gradient_clip,
    )


if __name__ == "__main__":
    main()
