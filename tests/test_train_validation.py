import unittest

import torch

from train.data import build_vocabs_from_records
from train.data import encode_records
from train.validation import evaluate_decode_cer
from train.validation import unwrap_subset


class EmptyDecodeModel:
    def __call__(self, x, y):
        vocab_size = 8
        logits = torch.full((1, x.shape[1], y.shape[1], vocab_size), -100.0)
        logits[..., 1] = 100.0
        return logits

    def eval(self):
        return self

    def train(self):
        return self


class TrainValidationTest(unittest.TestCase):
    def test_unwrap_plain_dataset(self):
        dataset = [1, 2, 3]

        base, indexes = unwrap_subset(dataset)

        self.assertEqual(base, dataset)
        self.assertEqual(indexes, [0, 1, 2])

    def test_evaluate_decode_cer_none(self):
        vocabs = build_vocabs_from_records([{"input": "a", "target": "あ"}])
        dataset = encode_records([{"input": "a", "target": "あ"}], vocabs)

        result = evaluate_decode_cer(
            EmptyDecodeModel(),
            dataset,
            vocabs.output_vocab,
            decoder="none",
            max_samples=1,
        )

        self.assertIsNone(result)

    def test_evaluate_decode_cer_greedy(self):
        vocabs = build_vocabs_from_records([{"input": "a", "target": "あ"}])
        dataset = encode_records([{"input": "a", "target": "あ"}], vocabs)

        result = evaluate_decode_cer(
            EmptyDecodeModel(),
            dataset,
            vocabs.output_vocab,
            decoder="greedy",
            max_samples=1,
        )

        self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
