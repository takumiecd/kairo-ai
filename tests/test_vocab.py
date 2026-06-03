import unittest

from dataset.vocab import build_input_vocab
from dataset.vocab import build_output_bpe_vocab
from dataset.vocab import build_output_vocab


class VocabTest(unittest.TestCase):
    def test_input_vocab_round_trip(self):
        vocab = build_input_vocab(["git commit", "bagu"])

        ids = vocab.encode("git")

        self.assertEqual(vocab.decode(ids), "git")
        self.assertIn("<pad>", vocab.token_to_id)
        self.assertIn("<unk>", vocab.token_to_id)

    def test_output_vocab_has_rnnt_special_tokens(self):
        vocab = build_output_vocab(['git commit -m "バグを修正した"'])

        self.assertEqual(vocab.id_to_token[:4], ["<pad>", "<blank>", "<bos>", "<unk>"])
        self.assertIn("修", vocab.token_to_id)
        self.assertIn("g", vocab.token_to_id)

    def test_output_bpe_vocab_encodes_repeated_subwords(self):
        vocab = build_output_bpe_vocab(
            ["修正した", "確認した", "修正した"],
            vocab_size=20,
            min_frequency=2,
        )

        self.assertIn("した", vocab.token_to_id)
        ids = vocab.encode("修正した")

        self.assertLess(len(ids), len("修正した"))
        self.assertEqual(vocab.decode(ids), "修正した")


if __name__ == "__main__":
    unittest.main()
