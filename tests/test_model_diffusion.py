import random
import unittest

from model.diffusion import KairoDiffusionModel
from train.diffusion.data import build_diffusion_vocabs_from_records
from train.diffusion.data import collate_diffusion_batch
from train.diffusion.data import encode_diffusion_records
from train.diffusion.loss import compute_diffusion_loss


class DiffusionModelTest(unittest.TestCase):
    def test_forward_and_loss_support_empty_context(self):
        records = [{"input": "honwokudasai", "target": "本を下さい。"}]
        vocabs = build_diffusion_vocabs_from_records(records)
        dataset = encode_diffusion_records(records, vocabs, max_positions=32)
        batch = collate_diffusion_batch(
            dataset.examples, vocabs, diffusion_steps=4, rng=random.Random(0)
        )
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
            diffusion_steps=4,
        )

        loss = compute_diffusion_loss(model, batch)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(model.token_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
