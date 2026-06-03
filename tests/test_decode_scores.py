import unittest

import torch

from decode.scores import Candidate
from decode.scores import normalize_candidate_confidence
from decode.scores import top_k_token_probs


class DecodeScoresTest(unittest.TestCase):
    def test_normalizes_candidate_scores(self):
        candidates = normalize_candidate_confidence(
            [
                Candidate(text="修正した", score=-1.0),
                Candidate(text="修整した", score=-3.0),
            ]
        )

        self.assertAlmostEqual(sum(candidate.confidence for candidate in candidates), 1.0)
        self.assertGreater(candidates[0].confidence, candidates[1].confidence)

    def test_top_k_token_probs(self):
        tokens = ["<blank>", "修", "正"]

        top = top_k_token_probs(torch.tensor([0.0, 2.0, 1.0]), tokens, k=2)

        self.assertEqual(top[0][0], "修")
        self.assertEqual(len(top), 2)


if __name__ == "__main__":
    unittest.main()
