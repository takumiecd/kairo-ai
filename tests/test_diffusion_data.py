import random
import unittest

from train.diffusion.data import build_diffusion_vocabs_from_records
from train.diffusion.data import collate_diffusion_batch
from train.diffusion.data import corrupt_tokens
from train.diffusion.data import encode_diffusion_records
from train.diffusion.data import mask_id


class DiffusionDataTest(unittest.TestCase):
    def test_max_timestep_masks_all_tokens(self):
        records = [{"input": "honwokudasai", "target": "本を下さい"}]
        vocabs = build_diffusion_vocabs_from_records(records)
        target_ids = vocabs.output_vocab.encode(records[0]["target"])

        noisy = corrupt_tokens(
            target_ids,
            timestep=8,
            diffusion_steps=8,
            mask_token_id=mask_id(vocabs),
            random_token_ids=target_ids,
            rng=random.Random(0),
        )

        self.assertEqual(noisy, [mask_id(vocabs)] * len(target_ids))

    def test_collate_supports_optional_context(self):
        records = [
            {"input": "honwokudasai", "context": "私は学生です。", "target": "本を下さい。"},
            {"input": "arigatou", "target": "ありがとう"},
        ]
        vocabs = build_diffusion_vocabs_from_records(records)
        dataset = encode_diffusion_records(records, vocabs, max_positions=64)

        batch = collate_diffusion_batch(
            dataset.examples, vocabs, diffusion_steps=4, rng=random.Random(0)
        )

        self.assertEqual(batch["inputs"].shape[0], 2)
        self.assertEqual(batch["contexts"].shape[0], 2)
        self.assertEqual(batch["noisy_canvas"].shape, batch["token_targets"].shape)
        self.assertEqual(batch["length_targets"].tolist(), [6, 5])


if __name__ == "__main__":
    unittest.main()
