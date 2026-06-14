import json
import tempfile
import unittest
from pathlib import Path

import torch

from decode.greedy import load_model_from_artifact
from model.transducer import KairoTransducer
from train.common.checkpoint import save_vocabs
from train.common.checkpoint import write_json
from train.rnnt.data import load_train_valid_datasets_and_vocabs
from train.rnnt.lora import main as lora_main

RECORDS = [
    {"input": "neko", "target": "猫"},
    {"input": "inu", "target": "犬"},
    {"input": "watashi", "target": "私"},
    {"input": "hon", "target": "本"},
]


def build_base_artifact(base_dir: Path, data_path: Path) -> dict:
    _dataset, _valid, vocabs = load_train_valid_datasets_and_vocabs(
        data_path, None, output_vocab_size=64, output_min_token_frequency=1
    )
    config = {
        "embed_dim": 8,
        "hidden_dim": 16,
        "input_embed_dim": 8,
        "output_embed_dim": 8,
        "encoder_hidden_dim": 16,
        "prediction_hidden_dim": 16,
        "joint_hidden_dim": 16,
        "encoder_type": "lstm",
        "prediction_type": "lstm",
        "encoder_layers": 1,
        "prediction_layers": 1,
        "num_heads": 4,
        "feedforward_dim": None,
        "dropout": 0.1,
        "max_positions": 256,
        "output_tokenizer": "char",
        "output_vocab_size": 64,
        "output_min_token_frequency": 1,
    }
    model = KairoTransducer(
        input_vocab_size=len(vocabs.input_vocab.id_to_token),
        output_vocab_size=len(vocabs.output_vocab.id_to_token),
        input_embed_dim=8,
        output_embed_dim=8,
        encoder_hidden_dim=16,
        prediction_hidden_dim=16,
        joint_hidden_dim=16,
        encoder_type="lstm",
        prediction_type="lstm",
        encoder_layers=1,
        prediction_layers=1,
        input_pad_id=vocabs.input_vocab.token_to_id["<pad>"],
        output_pad_id=vocabs.output_vocab.token_to_id["<pad>"],
    )
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_vocabs(base_dir, vocabs)
    write_json(base_dir / "config.json", config)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        base_dir / "checkpoints" / "best.pt",
    )
    return config


class TrainLoraSmokeTest(unittest.TestCase):
    def test_lora_run_exports_adapter_and_loadable_personalized_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "records.jsonl"
            data_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in RECORDS) + "\n",
                encoding="utf-8",
            )
            base_dir = tmp / "base"
            build_base_artifact(base_dir, data_path)

            out_dir = tmp / "out"
            lora_main(
                [
                    "--base-artifact-dir", str(base_dir),
                    "--data", str(data_path),
                    "--output-dir", str(out_dir),
                    "--epochs", "1",
                    "--batch-size", "2",
                    "--device", "cpu",
                    "--lora-rank", "4",
                    "--max-len", "64",
                ]
            )

            self.assertTrue((out_dir / "lora_adapter.pt").exists())
            personalized = out_dir / "personalized"
            self.assertTrue((personalized / "checkpoints" / "best.pt").exists())

            # The merged checkpoint must load into a plain KairoTransducer.
            model, _input_vocab, _output_vocab = load_model_from_artifact(personalized)
            x = torch.randint(0, len(_input_vocab.id_to_token), (1, 5))
            y = torch.randint(0, len(_output_vocab.id_to_token), (1, 3))
            logits = model(x, y)
            self.assertEqual(logits.shape[0], 1)

            # The adapter file holds only LoRA tensors plus config.
            adapter = torch.load(out_dir / "lora_adapter.pt", map_location="cpu")
            self.assertTrue(adapter["lora"])
            self.assertTrue(all("lora_" in key for key in adapter["lora"]))
            self.assertEqual(adapter["lora_config"]["rank"], 4)


if __name__ == "__main__":
    unittest.main()
