import unittest

import torch
from torch.utils.data import DataLoader

from model.transducer import KairoTransducer
from train.data import build_vocabs_from_records
from train.data import collate_transducer_batch
from train.data import encode_records
from train.overfit import compute_loss
from train.overfit import evaluate_average_loss


class TrainOverfitTest(unittest.TestCase):
    def test_evaluates_average_loss(self):
        records = [
            {"input": "shita", "target": "した"},
            {"input": "naoshita", "target": "直した"},
        ]
        vocabs = build_vocabs_from_records(records)
        dataset = encode_records(records, vocabs)
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=lambda examples: collate_transducer_batch(examples, vocabs),
        )
        model = KairoTransducer(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            embed_dim=8,
            hidden_dim=16,
        )

        loss = evaluate_average_loss(model, loader, vocabs.blank_id)

        self.assertGreater(loss, 0.0)

    def test_compute_loss_backpropagates(self):
        records = [{"input": "shita", "target": "した"}]
        vocabs = build_vocabs_from_records(records)
        dataset = encode_records(records, vocabs)
        batch = collate_transducer_batch([dataset[0]], vocabs)
        model = KairoTransducer(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            embed_dim=8,
            hidden_dim=16,
        )

        loss = compute_loss(model, batch, vocabs.blank_id)
        loss.backward()

        grad_norm = torch.norm(model.joint_fc2.weight.grad)
        self.assertGreater(float(grad_norm.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
