"""共通学習エンジン。

モデル非依存の部分（device 選択 / データ分割 / epoch ループ / AMP /
勾配クリップ / チェックポイント / metrics ログ / 曲線プロット / resume）を
ここへ集約する。各モデルの train エントリは「固有 arg・データ・モデル・
損失クロージャ」だけ用意して `Trainer.fit(...)` を呼ぶ薄い殻になる。

モデル固有のもの（RNN-T の dims 推論など）はここには置かない。
損失は ``loss_fn(model, batch) -> Tensor``、検証 CER は ``cer_fn(epoch) ->
float | None`` のクロージャで受け取り、エンジンはモデルの中身を知らない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from train.checkpoint import load_checkpoint
from train.checkpoint import save_checkpoint
from train.loss import move_batch_to_device


# ----------------------------------------------------------------------
# device / データ
# ----------------------------------------------------------------------
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


def build_loader(dataset, collate_fn, batch_size: int, shuffle: bool) -> DataLoader:
    """collate を引数で受ける汎用ローダ（モデルごとの collate を渡す）。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


# ----------------------------------------------------------------------
# 共通 argparse
# ----------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """全モデル共通の学習フラグを登録する。"""
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from.")
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
        "--output-tokenizer",
        choices=["char", "bpe"],
        default="char",
        help="Tokenizer for output targets. Input remains character-tokenized.",
    )
    parser.add_argument("--output-vocab-size", type=int, default=4000)
    parser.add_argument("--output-min-token-frequency", type=int, default=2)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for training (CUDA only).",
    )


# ----------------------------------------------------------------------
# resume / metrics
# ----------------------------------------------------------------------
def load_best_valid_loss(output_dir: Path) -> float:
    best_path = Path(output_dir) / "checkpoints" / "best.pt"
    if not best_path.exists():
        return float("inf")
    state = load_checkpoint(best_path, map_location="cpu")
    return float(state.get("valid_loss", float("inf")))


def restore_training_state(model, optimizer, checkpoint: dict) -> int:
    """resume: model/optimizer を復元し、次の開始 epoch を返す。"""
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"]) + 1


def trim_metrics_log(output_dir: Path, start_epoch: int) -> None:
    """``start_epoch`` 以降の古い行を削って曲線が連続するようにする。

    ``metrics.jsonl`` は追記専用で ``plot_metrics`` はファイル順に描く。
    resume（特に best.pt のような過去チェックポイント）時に epoch >=
    start_epoch の古い行が残ると、曲線が後ろに飛んで再スタートに見える。
    """
    metrics_log_path = Path(output_dir) / "metrics.jsonl"
    if not metrics_log_path.exists():
        return
    kept: list[str] = []
    with metrics_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("epoch", 0) < start_epoch:
                kept.append(line if line.endswith("\n") else line + "\n")
    with metrics_log_path.open("w", encoding="utf-8") as f:
        f.writelines(kept)


def plot_metrics(output_dir: Path) -> None:
    metrics_log_path = Path(output_dir) / "metrics.jsonl"
    if not metrics_log_path.exists():
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = []
        train_losses = []
        valid_losses = []
        valid_cers = []
        cer_epochs = []

        with open(metrics_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    epochs.append(data["epoch"])
                    train_losses.append(data["train_loss"])
                    valid_losses.append(data["valid_loss"])
                    if "valid_cer" in data and data["valid_cer"] is not None:
                        valid_cers.append(data["valid_cer"])
                        cer_epochs.append(data["epoch"])

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Train Loss", marker="o", color="blue")
        plt.plot(epochs, valid_losses, label="Validation Loss", marker="x", color="red")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        loss_output_path = Path(output_dir) / "loss_curve.png"
        plt.savefig(loss_output_path, dpi=150, bbox_inches="tight")
        plt.close()

        if valid_cers:
            plt.figure(figsize=(10, 6))
            plt.plot(cer_epochs, valid_cers, label="Validation CER", marker="s", color="green")
            plt.title("Validation Character Error Rate (CER)")
            plt.xlabel("Epoch")
            plt.ylabel("CER")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()

            cer_output_path = Path(output_dir) / "cer_curve.png"
            plt.savefig(cer_output_path, dpi=150, bbox_inches="tight")
            plt.close()

    except ImportError:
        pass


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
LossFn = Callable[[object, dict], torch.Tensor]
ValidLossFn = Callable[[], float]
CerFn = Callable[[int], "float | None"]


class Trainer:
    """モデル非依存の epoch ループ + チェックポイント + metrics。"""

    def __init__(
        self,
        *,
        model,
        optimizer,
        device: torch.device,
        output_dir: Path,
        config,
        amp: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.amp_enabled = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: LossFn,
        gradient_clip: float,
        epoch: int,
        epochs: int,
    ) -> float:
        print(f"Epoch {epoch}/{epochs} started. Training on {len(train_loader)} batches...")
        self.model.train()
        train_losses: list[float] = []

        try:
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
            has_tqdm = True
        except ImportError:
            pbar = train_loader
            has_tqdm = False

        for i, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                loss = loss_fn(self.model, batch)
            self.scaler.scale(loss).backward()
            if gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = float(loss.item())
            train_losses.append(loss_val)
            if has_tqdm:
                pbar.set_postfix(loss=f"{loss_val:.4f}")
            elif i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}: loss={loss_val:.4f}")

        return sum(train_losses) / len(train_losses)

    def _checkpoint_and_log(
        self,
        epoch: int,
        train_loss: float,
        valid_loss: float,
        valid_cer: "float | None",
        best_valid_loss: float,
    ) -> float:
        metrics = [
            f"epoch={epoch}",
            f"train_loss={train_loss:.4f}",
            f"valid_loss={valid_loss:.4f}",
        ]
        if valid_cer is not None:
            metrics.append(f"valid_cer={valid_cer:.4f}")
        print(" ".join(metrics))

        save_checkpoint(
            self.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
            self.model,
            self.optimizer,
            epoch,
            train_loss,
            valid_loss,
            self.config,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                self.output_dir / "checkpoints" / "best.pt",
                self.model,
                self.optimizer,
                epoch,
                train_loss,
                valid_loss,
                self.config,
            )

        metrics_record = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        if valid_cer is not None:
            metrics_record["valid_cer"] = valid_cer
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_record) + "\n")
        plot_metrics(self.output_dir)
        return best_valid_loss

    def fit(
        self,
        *,
        train_loader: DataLoader,
        epochs: int,
        loss_fn: LossFn,
        start_epoch: int = 1,
        valid_loss_fn: ValidLossFn | None = None,
        cer_fn: CerFn | None = None,
        gradient_clip: float = 1.0,
    ) -> None:
        best_valid_loss = load_best_valid_loss(self.output_dir)
        trim_metrics_log(self.output_dir, start_epoch)
        for epoch in range(start_epoch, epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader, loss_fn, gradient_clip, epoch, epochs
            )
            valid_loss = valid_loss_fn() if valid_loss_fn is not None else train_loss
            valid_cer = cer_fn(epoch) if cer_fn is not None else None
            best_valid_loss = self._checkpoint_and_log(
                epoch, train_loss, valid_loss, valid_cer, best_valid_loss
            )
