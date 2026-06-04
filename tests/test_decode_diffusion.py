import unittest

import torch

from decode.diffusion import diffusion_decode
from model.diffusion import KairoDiffusionModel
from train.diffusion.data import build_diffusion_vocabs_from_records


class DiffusionDecodeTest(unittest.TestCase):
    def test_decode_returns_requested_length_without_special_tokens(self):
        records = [{"input": "honwokudasai", "target": "本を下さい。"}]
        vocabs = build_diffusion_vocabs_from_records(records)
        model = KairoDiffusionModel(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            model_dim=16,
            input_embed_dim=8,
            output_embed_dim=8,
            num_heads=2,
            num_input_layers=1,
            num_context_layers=1,
            num_canvas_layers=1,
            feedforward_dim=32,
            dropout=0.0,
            max_positions=32,
            diffusion_steps=2,
        )
        with torch.no_grad():
            model.token_head.weight.zero_()
            model.token_head.bias.fill_(-10.0)
            model.token_head.bias[vocabs.output_vocab.token_to_id["本"]] = 10.0

        decoded = diffusion_decode(
            model,
            vocabs.input_vocab.encode("honwokudasai"),
            [],
            vocabs.output_vocab,
            output_length=3,
        )

        self.assertEqual(decoded, "本本本")


if __name__ == "__main__":
    unittest.main()
