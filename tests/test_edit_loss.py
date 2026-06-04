import unittest

from model.edit_transducer import KairoEditTransducer
from train.common.data import build_vocabs_from_records
from train.edit.data import collate_edit_batch
from train.edit.data import encode_edit_example
from train.edit.loss import compute_edit_loss


class EditLossTest(unittest.TestCase):
    def test_compute_edit_loss_returns_trainable_loss(self):
        records = [{"input": "kyouhahashide", "target": "今日は箸で"}]
        vocabs = build_vocabs_from_records(records)
        example = encode_edit_example("kyouhahashide", "今日は橋で", "今日は箸で", vocabs)
        batch = collate_edit_batch([example], vocabs)
        model = KairoEditTransducer(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            input_embed_dim=8,
            output_embed_dim=8,
            action_embed_dim=4,
            hidden_dim=16,
        )

        loss = compute_edit_loss(model, batch)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
