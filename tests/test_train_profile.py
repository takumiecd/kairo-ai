import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.transducer import KairoTransducer
from train.rnnt.loss import compute_rnnt_loss_with_profile
from train.rnnt.loss import evaluate_average_loss
from train.rnnt.profile_data import collate_profile_transducer_batch
from train.rnnt.profile_data import load_profile_dataset_and_vocabs


def _write_profile_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sample_records() -> list[dict]:
    """3件の擬似プロファイル付きレコード。u_m が育つ様子を模した profile 付き。"""
    empty_profile = {
        "meta": {"version": 1, "base_profile_id": None, "total_units": 0},
        "explicit": [],
        "implicit": {
            "unigram": {},
            "recency": [],
            "domain": {"code": 1 / 3, "prose": 1 / 3, "chat": 1 / 3},
            "lang": {"ja_ratio": 0.0, "en_token_rate": 0.0},
        },
    }
    grown_profile = {
        "meta": {"version": 1, "base_profile_id": None, "total_units": 4},
        "explicit": [],
        "implicit": {
            "unigram": {"猫": {"count": 1.0, "reading": None, "last_used": 2}},
            "recency": ["猫"],
            "domain": {"code": 0.0, "prose": 1.0, "chat": 0.0},
            "lang": {"ja_ratio": 1.0, "en_token_rate": 0.0},
        },
    }
    return [
        {"persona": "p", "input": "shita", "context": "", "profile": empty_profile, "target": "した"},
        {"persona": "p", "input": "naoshita", "context": "した", "profile": grown_profile, "target": "直した"},
        {"persona": "p", "input": "mita", "context": "直した", "profile": grown_profile, "target": "見た"},
    ]


class ProfileDataLoadingTest(unittest.TestCase):
    def test_load_profile_dataset_and_vocabs_reads_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.jsonl"
            _write_profile_jsonl(path, _sample_records())
            dataset, vocabs = load_profile_dataset_and_vocabs(path)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0].target_text, "した")
        self.assertIn("total_units", dataset[1].profile["meta"])

    def test_collate_produces_profile_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.jsonl"
            _write_profile_jsonl(path, _sample_records())
            dataset, vocabs = load_profile_dataset_and_vocabs(path)

        batch = collate_profile_transducer_batch(
            [dataset[index] for index in range(len(dataset))],
            vocabs,
            top_k=8,
        )
        self.assertEqual(batch["profile_domain"].shape, (3, 3))
        self.assertEqual(batch["profile_lang"].shape, (3, 2))
        self.assertEqual(batch["profile_word_mask"].shape[0], 3)
        # 標準の RNN-T バッチキーもそのまま含む。
        self.assertIn("inputs", batch)
        self.assertIn("prediction_inputs", batch)

    def test_profile_drop_replaces_with_empty_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.jsonl"
            _write_profile_jsonl(path, _sample_records())
            dataset, vocabs = load_profile_dataset_and_vocabs(path)

        examples = [dataset[1]]  # grown_profile を持つ例
        rng = random.Random(0)
        # drop_rate=1.0 なら必ず空プロファイルに置換される -> word_mask が全て False。
        batch = collate_profile_transducer_batch(
            examples, vocabs, top_k=8, profile_drop_rate=1.0, rng=rng
        )
        self.assertEqual(int(batch["profile_word_mask"].sum().item()), 0)

    def test_profile_drop_zero_keeps_original_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.jsonl"
            _write_profile_jsonl(path, _sample_records())
            dataset, vocabs = load_profile_dataset_and_vocabs(path)

        examples = [dataset[1]]
        batch = collate_profile_transducer_batch(
            examples, vocabs, top_k=8, profile_drop_rate=0.0
        )
        self.assertGreater(int(batch["profile_word_mask"].sum().item()), 0)


class ProfileConditionedLossTest(unittest.TestCase):
    def test_compute_rnnt_loss_with_profile_backpropagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.jsonl"
            _write_profile_jsonl(path, _sample_records())
            dataset, vocabs = load_profile_dataset_and_vocabs(path)

        batch = collate_profile_transducer_batch(
            [dataset[index] for index in range(len(dataset))], vocabs, top_k=8
        )
        model = KairoTransducer(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            embed_dim=8,
            hidden_dim=16,
            profile_conditioning=True,
            profile_top_k=8,
        )

        loss = compute_rnnt_loss_with_profile(model, batch, vocabs.blank_id)
        loss.backward()

        grad_norm = torch.norm(model.profile_encoder.mlp[0].weight.grad)
        self.assertGreater(float(grad_norm.item()), 0.0)


class ProfileOverfitSmokeTest(unittest.TestCase):
    """train.rnnt.overfit 相当の、段階B条件付け版の極小 smoke test。

    数十 step で loss が下がることだけを確認する(CPU で 1 分以内)。
    """

    def test_loss_decreases_after_training_steps(self):
        records = _sample_records() * 4  # 12 件に増やして最低限のバリエーションを持たせる
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.jsonl"
            _write_profile_jsonl(path, records)
            dataset, vocabs = load_profile_dataset_and_vocabs(path)

        torch.manual_seed(0)
        model = KairoTransducer(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            embed_dim=16,
            hidden_dim=32,
            profile_conditioning=True,
            profile_top_k=8,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

        rng = random.Random(0)
        collate = lambda examples: collate_profile_transducer_batch(
            examples, vocabs, top_k=8, profile_drop_rate=0.3, rng=rng
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
        eval_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)

        initial_loss = evaluate_average_loss(
            model, eval_loader, vocabs.blank_id, loss_fn=compute_rnnt_loss_with_profile
        )

        iterator = iter(loader)
        for _ in range(40):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_rnnt_loss_with_profile(model, batch, vocabs.blank_id)
            loss.backward()
            optimizer.step()

        final_loss = evaluate_average_loss(
            model, eval_loader, vocabs.blank_id, loss_fn=compute_rnnt_loss_with_profile
        )

        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
