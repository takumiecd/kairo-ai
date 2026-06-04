"""Train the Kairo iterative edit refiner on JSONL data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Subset

from eval.metrics import mean_cer
from model.edit_refiner import KairoEditRefiner
from train.common.checkpoint import load_checkpoint
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.common.data import TrainingVocabs
from train.refiner.data import collate_refine_batch
from train.refiner.data import load_train_valid_refine_datasets_and_vocabs
from train.refiner.data import placeholder_id
from train.refiner.loss import compute_refine_loss
from train.refiner.loss import evaluate_average_refine_loss
from train.common.engine import Trainer
from train.common.engine import add_common_args
from train.common.engine import build_loader
from train.common.engine import restore_training_state
from train.common.engine import select_device
from train.common.engine import split_dataset
from train.common.validation import unwrap_subset


@dataclass(frozen=True)
class RefineTrainConfig:
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
    num_hypothesis_layers: int
    feedforward_dim: int
    dropout: float
    max_insertions_per_gap: int
    max_positions: int
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    limit_examples: int | None
    gradient_clip: float
    insert_loss_weight: float
    fill_loss_weight: float
    valid_decode: str
    valid_cer_samples: int
    valid_cer_every: int
    valid_max_rounds: int
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
    parser.add_argument("--num-hypothesis-layers", type=int, default=3)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-insertions-per-gap", type=int, default=8)
    parser.add_argument(
        "--max-positions",
        type=int,
        default=512,
        help="Positional-embedding table size. Examples longer than this are dropped.",
    )
    parser.add_argument("--insert-loss-weight", type=float, default=1.0)
    parser.add_argument("--fill-loss-weight", type=float, default=1.0)
    parser.add_argument("--valid-max-rounds", type=int, default=2)
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
    parser.add_argument(
        "--vocab-sample",
        type=int,
        default=None,
        help="Build the BPE vocab from this many randomly sampled records "
        "(base chars still cover all data). Speeds up vocab building on large corpora.",
    )
    return parser.parse_args()


def decode_refined(ids: list[int], vocabs: TrainingVocabs) -> str:
    """精製後の id 列を本文へ。BOS/EOS/PAD/<plh> は落とす。"""
    drop = {
        vocabs.output_vocab.token_to_id["<bos>"],
        placeholder_id(vocabs),
        vocabs.output_pad_id,
    }
    eos = vocabs.output_vocab.token_to_id["<eos>"]
    out: list[int] = []
    for token_id in ids:
        if token_id == eos:
            break
        if token_id not in drop:
            out.append(token_id)
    return vocabs.output_vocab.decode(out)


@torch.no_grad()
def evaluate_refine_cer(model, valid_dataset, vocabs, max_samples, max_rounds) -> float:
    base_dataset, indexes = unwrap_subset(valid_dataset)
    selected = indexes[:max_samples]
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    for index in selected:
        example = base_dataset[index]
        refined = model.refine(example.input_ids, example.hypothesis_ids, max_rounds=max_rounds)
        predictions.append(decode_refined(refined, vocabs))
        references.append(example.target_text)
    model.train()
    return mean_cer(predictions, references)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)

    # 語彙の再利用: --vocab 明示、または resume 時はその run の output_dir を流用
    # （再構築せず checkpoint と同じ語彙を保証＆BPE 構築を省略）。
    vocab_dir = args.vocab
    if vocab_dir is None and args.resume is not None:
        vocab_dir = args.output_dir
    cache_dir = None if args.no_cache else (args.cache_dir or Path(args.data).resolve().parent / ".kairo_cache")

    dataset, explicit_valid_dataset, vocabs = load_train_valid_refine_datasets_and_vocabs(
        args.data,
        args.valid_data,
        output_tokenizer=args.output_tokenizer,
        output_vocab_size=args.output_vocab_size,
        output_min_token_frequency=args.output_min_token_frequency,
        max_insertions_per_gap=args.max_insertions_per_gap,
        vocab_dir=vocab_dir,
        cache_dir=cache_dir,
        vocab_sample=args.vocab_sample,
        vocab_sample_seed=args.seed,
        max_positions=args.max_positions,
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

    collate = lambda examples: collate_refine_batch(examples, vocabs)
    train_loader = build_loader(train_dataset, collate, args.batch_size, shuffle=True)
    valid_loader = (
        build_loader(valid_dataset, collate, args.batch_size, shuffle=False)
        if len(valid_dataset) > 0
        else None
    )

    config = RefineTrainConfig(
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
        num_hypothesis_layers=args.num_hypothesis_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_insertions_per_gap=args.max_insertions_per_gap,
        max_positions=args.max_positions,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        limit_examples=args.limit_examples,
        gradient_clip=args.gradient_clip,
        insert_loss_weight=args.insert_loss_weight,
        fill_loss_weight=args.fill_loss_weight,
        valid_decode=args.valid_decode,
        valid_cer_samples=args.valid_cer_samples,
        valid_cer_every=args.valid_cer_every,
        valid_max_rounds=args.valid_max_rounds,
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

    model = KairoEditRefiner(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        placeholder_id=placeholder_id(vocabs),
        model_dim=args.model_dim,
        input_embed_dim=args.input_embed_dim,
        output_embed_dim=args.output_embed_dim,
        num_heads=args.num_heads,
        num_input_layers=args.num_input_layers,
        num_hypothesis_layers=args.num_hypothesis_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_insertions_per_gap=args.max_insertions_per_gap,
        max_positions=args.max_positions,
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
            or len(valid_dataset) == 0
            or args.valid_decode == "none"
            or args.valid_cer_every <= 0
            or epoch % args.valid_cer_every != 0
        ):
            return None
        return evaluate_refine_cer(
            model,
            valid_dataset,
            vocabs,
            max_samples=args.valid_cer_samples,
            max_rounds=args.valid_max_rounds,
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
        loss_fn=lambda m, batch: compute_refine_loss(
            m, batch, args.insert_loss_weight, args.fill_loss_weight
        ),
        start_epoch=start_epoch,
        valid_loss_fn=(
            (
                lambda: evaluate_average_refine_loss(
                    model,
                    valid_loader,
                    device=device,
                    insert_loss_weight=args.insert_loss_weight,
                    fill_loss_weight=args.fill_loss_weight,
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
