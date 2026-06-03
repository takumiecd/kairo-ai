import unittest
import tempfile
from pathlib import Path
from dataclasses import dataclass

import torch

from train.checkpoint import load_checkpoint
from train.checkpoint import save_checkpoint
from train.train import select_device
from train.train import split_dataset
from train.train import load_best_valid_loss


class TrainEntrypointTest(unittest.TestCase):
    def test_selects_cpu_device(self):
        self.assertEqual(str(select_device("cpu")), "cpu")

    def test_split_dataset_creates_validation_subset(self):
        dataset = list(range(10))

        train_dataset, valid_dataset = split_dataset(dataset, validation_ratio=0.2, seed=0)

        self.assertEqual(len(train_dataset), 8)
        self.assertEqual(len(valid_dataset), 2)

    def test_split_dataset_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            split_dataset([1, 2, 3], validation_ratio=1.0, seed=0)

    def test_load_best_valid_loss_defaults_to_inf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_best_valid_loss(Path(tmpdir)), float("inf"))

    def test_checkpoint_round_trip_has_epoch(self):
        @dataclass(frozen=True)
        class Config:
            name: str = "test"

        config = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters())

            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=3,
                train_loss=2.0,
                valid_loss=1.0,
                config=config,
            )
            state = load_checkpoint(path, map_location="cpu")

        self.assertEqual(state["epoch"], 3)
        self.assertEqual(state["valid_loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
