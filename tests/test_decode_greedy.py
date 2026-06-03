import json
import tempfile
import unittest
from pathlib import Path

import torch

from dataset.vocab import build_input_vocab
from dataset.vocab import build_output_vocab
from dataset.vocab import vocab_from_token_to_id
from decode.greedy import greedy_decode
from decode.greedy import load_vocab


class DummyGreedyModel:
    def __init__(self, token_ids: list[int], blank_id: int) -> None:
        self.token_ids = token_ids
        self.blank_id = blank_id

    def __call__(self, x, y):
        vocab_size = max(self.token_ids + [self.blank_id]) + 1
        logits = torch.full((1, x.shape[1], y.shape[1], vocab_size), -100.0)
        emitted_count = y.shape[1] - 1
        if emitted_count < len(self.token_ids):
            token_id = self.token_ids[emitted_count]
        else:
            token_id = self.blank_id
        logits[0, :, y.shape[1] - 1, token_id] = 100.0
        return logits


class DecodeGreedyTest(unittest.TestCase):
    def test_vocab_round_trip_from_json(self):
        vocab = build_input_vocab(["abc"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            path.write_text(json.dumps(vocab.to_dict()), encoding="utf-8")

            loaded = load_vocab(path)

        self.assertEqual(loaded.encode("abc"), vocab.encode("abc"))

    def test_vocab_from_token_to_id(self):
        vocab = vocab_from_token_to_id({"<unk>": 1, "<pad>": 0, "a": 2})

        self.assertEqual(vocab.id_to_token, ["<pad>", "<unk>", "a"])

    def test_greedy_decode_emits_until_blank(self):
        output_vocab = build_output_vocab(["した"])
        token_ids = output_vocab.encode("した")
        model = DummyGreedyModel(token_ids, output_vocab.token_to_id["<blank>"])

        decoded = greedy_decode(
            model,
            input_ids=[2, 3, 4],
            output_vocab=output_vocab,
            max_symbols_per_step=4,
        )

        self.assertEqual(decoded, "した")


if __name__ == "__main__":
    unittest.main()
