"""LoRA personalization fine-tune for the Kairo RNN-T.

Starts from a published base artifact (config + vocab + checkpoint), injects
LoRA adapters into the joint network, freezes the base, and fine-tunes only the
adapters on confirmed-feedback records (produced by ``dataset.source_feedback``).

Exports two things to ``--output-dir``:

- ``lora_adapter.pt`` -- the small portable adapter (LoRA tensors + config);
- ``personalized/`` -- a full artifact dir whose checkpoint has the LoRA deltas
  merged into the base weights, so the existing decoders / IME server can use it
  with no adapter-aware code (point ``--artifact-dir`` at it).

Run, e.g.::

    python -m train.rnnt.lora \
      --base-artifact-dir artifacts/rnnt-trf-v2 \
      --data data/feedback/records.jsonl \
      --output-dir artifacts/rnnt-trf-v2-me \
      --epochs 3 --batch-size 16 --learning-rate 1e-3 \
      --lora-rank 8 --lora-alpha 16
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from decode.greedy import load_model_from_artifact
from model.lora import count_trainable_parameters
from model.lora import inject_lora
from model.lora import lora_state_dict
from model.lora import mark_only_lora_trainable
from model.lora import merged_state_dict
from train.common.batch import move_batch_to_device
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.common.engine import build_loader
from train.common.engine import select_device
from train.rnnt.data import collate_transducer_batch
from train.rnnt.data import load_train_valid_datasets_and_vocabs
from train.rnnt.loss import compute_rnnt_loss


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint inside the base artifact (default: checkpoints/best.pt).",
    )
    parser.add_argument("--data", type=Path, required=True, help="Feedback records JSONL (input/target).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="joint_fc1,joint_fc2",
        help="Comma-separated substrings of linear-layer names to adapt.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)

    base_dir = args.base_artifact_dir
    with (base_dir / "config.json").open("r", encoding="utf-8") as file:
        base_config = json.load(file)
    checkpoint_path = args.base_checkpoint or base_dir / "checkpoints" / "best.pt"

    model, _input_vocab, _output_vocab = load_model_from_artifact(base_dir, checkpoint_path)
    targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    replaced = inject_lora(
        model,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=targets,
    )
    if not replaced:
        raise SystemExit(f"no linear layers matched --lora-targets={args.lora_targets!r}")
    mark_only_lora_trainable(model)
    model.to(device)
    print(
        f"lora_targets={replaced} rank={args.lora_rank} alpha={args.lora_alpha} "
        f"trainable_params={count_trainable_parameters(model)}",
        flush=True,
    )

    dataset, _valid, vocabs = load_train_valid_datasets_and_vocabs(
        args.data,
        None,
        max_len=args.max_len,
        output_tokenizer=base_config.get("output_tokenizer", "char"),
        output_vocab_size=int(base_config.get("output_vocab_size", 4000)),
        output_min_token_frequency=int(base_config.get("output_min_token_frequency", 2)),
        vocab_dir=base_dir,
    )
    if len(dataset) == 0:
        raise SystemExit(f"no training records in {args.data}")

    collate = lambda examples: collate_transducer_batch(examples, vocabs)
    loader = build_loader(dataset, collate, args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = compute_rnnt_loss(model, batch, vocabs.blank_id)
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.gradient_clip
                )
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        print(f"epoch={epoch} train_loss={total / max(batches, 1):.4f}", flush=True)

    export(args.output_dir, model, vocabs, base_config, base_dir, checkpoint_path, args, targets)


def export(output_dir, model, vocabs, base_config, base_dir, checkpoint_path, args, targets) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = {
        "lora": lora_state_dict(model),
        "lora_config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": list(targets),
        },
        "base_artifact_dir": str(base_dir),
        "base_checkpoint": str(checkpoint_path),
    }
    torch.save(adapter, output_dir / "lora_adapter.pt")

    # A full artifact whose checkpoint has the adapter merged into the base, so
    # the existing decoders / IME server load it as a normal model.
    personalized = output_dir / "personalized"
    (personalized / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_json(personalized / "config.json", base_config)
    save_vocabs(personalized, vocabs)
    torch.save(
        {"model_state_dict": merged_state_dict(model), "config": base_config},
        personalized / "checkpoints" / "best.pt",
    )
    print(
        f"wrote {output_dir / 'lora_adapter.pt'} and personalized artifact {personalized}",
        flush=True,
    )


if __name__ == "__main__":
    main()
