import unittest

from train.data import build_vocabs_from_records
from train.data import collate_transducer_batch
from train.data import encode_records


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


if __name__ == "__main__":
    unittest.main()
