import unittest

from train.common.checkpoint import save_vocabs
from train.common.data import build_vocabs_from_records
from train.rnnt.data import collate_transducer_batch
from train.rnnt.data import encode_records
from train.rnnt.data import load_train_valid_datasets_and_vocabs
from dataset.split import write_jsonl


class TrainDataTest(unittest.TestCase):
    def test_collate_builds_rnnt_batch(self):
        records = [
            {
                "input": 'git commit -m "baguwoshuuseishita"',
                "target": 'git commit -m "バグを修正した"',
            },
            {
                "input": "src/main.rs wokakuninshita",
                "target": "src/main.rs を確認した",
            },
        ]
        vocabs = build_vocabs_from_records(records)
        dataset = encode_records(records, vocabs)

        batch = collate_transducer_batch([dataset[0], dataset[1]], vocabs)

        self.assertEqual(batch["inputs"].shape[0], 2)
        self.assertEqual(batch["targets"].shape[0], 2)
        self.assertEqual(batch["prediction_inputs"].shape[1], batch["targets"].shape[1] + 1)
        self.assertTrue((batch["prediction_inputs"][:, 0] == vocabs.bos_id).all())
        self.assertEqual(str(batch["targets"].dtype), "torch.int32")
        self.assertEqual(vocabs.blank_id, 1)
        self.assertEqual(dataset[0].input_text, 'git commit -m "baguwoshuuseishita"')
        self.assertEqual(dataset[0].target_text, 'git commit -m "バグを修正した"')

    def test_load_train_valid_datasets_builds_shared_vocab(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.jsonl"
            valid_path = Path(tmpdir) / "valid.jsonl"
            write_jsonl(train_path, [{"input": "abc", "target": "修正"}])
            write_jsonl(valid_path, [{"input": "xyz", "target": "確認"}])

            train_dataset, valid_dataset, vocabs = load_train_valid_datasets_and_vocabs(
                train_path,
                valid_path,
            )

        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(valid_dataset), 1)
        self.assertIn("確", vocabs.output_vocab.token_to_id)

    def test_load_train_valid_datasets_reuses_saved_vocab(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.jsonl"
            valid_path = tmp_path / "valid.jsonl"
            vocab_dir = tmp_path / "vocab"
            write_jsonl(train_path, [{"input": "abc", "target": "修正"}])
            write_jsonl(valid_path, [{"input": "xyz", "target": "確認"}])

            saved_vocabs = build_vocabs_from_records(
                [{"input": "abc", "target": "修正"}]
            )
            save_vocabs(vocab_dir, saved_vocabs)

            train_dataset, valid_dataset, vocabs = load_train_valid_datasets_and_vocabs(
                train_path,
                valid_path,
                vocab_dir=vocab_dir,
            )

        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(valid_dataset), 1)
        self.assertEqual(vocabs.output_vocab.id_to_token, saved_vocabs.output_vocab.id_to_token)
        self.assertNotIn("確", vocabs.output_vocab.token_to_id)
        self.assertIn(saved_vocabs.output_vocab.token_to_id["<unk>"], valid_dataset[0].target_ids)

    def test_build_vocabs_supports_bpe_output_tokenizer(self):
        records = [
            {"input": "shuuseishita", "target": "修正した"},
            {"input": "kakuninishita", "target": "確認した"},
            {"input": "moushikomishita", "target": "申し込みした"},
        ]

        vocabs = build_vocabs_from_records(
            records,
            output_tokenizer="bpe",
            output_vocab_size=30,
            output_min_token_frequency=2,
        )
        dataset = encode_records(records, vocabs)

        self.assertIn("した", vocabs.output_vocab.token_to_id)
        self.assertLess(len(dataset[0].target_ids), len(records[0]["target"]))


if __name__ == "__main__":
    unittest.main()
