import unittest

from dataset.vocab import build_input_vocab
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


if __name__ == "__main__":
    unittest.main()
