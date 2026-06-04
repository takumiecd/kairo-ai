import unittest
import tempfile
from pathlib import Path
from dataclasses import dataclass

import torch

from train.common.checkpoint import load_checkpoint
from train.common.checkpoint import save_checkpoint
from train.rnnt.train import select_device
from train.rnnt.train import split_dataset
from train.rnnt.train import load_best_valid_loss
from train.rnnt.train import get_resume_model_dims
from train.rnnt.train import ModelDims


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

    def test_resume_model_dims_prefers_checkpoint_config(self):
        checkpoint = {
            "config": {"embed_dim": 128, "hidden_dim": 256},
            "model_state_dict": {},
        }

        self.assertEqual(
            get_resume_model_dims(checkpoint),
            ModelDims(
                input_embed_dim=128,
                output_embed_dim=128,
                encoder_hidden_dim=256,
                prediction_hidden_dim=256,
                joint_hidden_dim=256,
            ),
        )

    def test_resume_model_dims_reads_split_checkpoint_config(self):
        checkpoint = {
            "config": {
                "input_embed_dim": 96,
                "output_embed_dim": 256,
                "encoder_hidden_dim": 512,
                "prediction_hidden_dim": 384,
                "joint_hidden_dim": 320,
            },
            "model_state_dict": {},
        }

        self.assertEqual(
            get_resume_model_dims(checkpoint),
            ModelDims(
                input_embed_dim=96,
                output_embed_dim=256,
                encoder_hidden_dim=512,
                prediction_hidden_dim=384,
                joint_hidden_dim=320,
            ),
        )

    def test_resume_model_dims_can_infer_from_state_dict(self):
        checkpoint = {
            "model_state_dict": {
                "encoder_emb.weight": torch.zeros(50, 128),
                "encoder_lstm.weight_hh_l0": torch.zeros(1024, 256),
                "pred_emb.weight": torch.zeros(50, 96),
                "pred_lstm.weight_hh_l0": torch.zeros(768, 192),
                "joint_fc1.weight": torch.zeros(320, 448),
            }
        }

        self.assertEqual(
            get_resume_model_dims(checkpoint),
            ModelDims(
                input_embed_dim=128,
                output_embed_dim=96,
                encoder_hidden_dim=256,
                prediction_hidden_dim=192,
                joint_hidden_dim=320,
            ),
        )


if __name__ == "__main__":
    unittest.main()
