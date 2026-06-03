import unittest

import torch

from dataset.vocab import build_output_vocab
from decode.beam import beam_search_decode


class DummyBeamModel:
    def __init__(self, first_id: int, second_id: int, blank_id: int) -> None:
        self.first_id = first_id
        self.second_id = second_id
        self.blank_id = blank_id

    def __call__(self, x, y):
        vocab_size = max(self.first_id, self.second_id, self.blank_id) + 1
        logits = torch.full((1, x.shape[1], y.shape[1], vocab_size), -100.0)
        emitted_count = y.shape[1] - 1
        if emitted_count == 0:
            logits[0, :, y.shape[1] - 1, self.first_id] = 8.0
            logits[0, :, y.shape[1] - 1, self.second_id] = 7.0
            logits[0, :, y.shape[1] - 1, self.blank_id] = 0.0
        else:
            logits[0, :, y.shape[1] - 1, self.blank_id] = 8.0
        return logits


class DecodeBeamTest(unittest.TestCase):
    def test_beam_search_returns_confidence_ranked_candidates(self):
        output_vocab = build_output_vocab(["した"])
        first_id = output_vocab.token_to_id["し"]
        second_id = output_vocab.token_to_id["た"]
        blank_id = output_vocab.token_to_id["<blank>"]
        model = DummyBeamModel(first_id, second_id, blank_id)

        candidates = beam_search_decode(
            model,
            input_ids=[2],
            output_vocab=output_vocab,
            beam_width=2,
            expansion_width=2,
            max_symbols_per_step=2,
        )

        self.assertEqual(candidates[0].text, "し")
        self.assertGreater(candidates[0].confidence, candidates[1].confidence)
        self.assertAlmostEqual(sum(candidate.confidence for candidate in candidates), 1.0)


if __name__ == "__main__":
    unittest.main()
