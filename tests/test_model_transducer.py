import unittest

from model.transducer import KairoTransducer


class KairoTransducerTest(unittest.TestCase):
    def test_supports_split_embedding_and_hidden_dims(self):
        model = KairoTransducer(
            input_vocab_size=10,
            output_vocab_size=20,
            input_embed_dim=8,
            output_embed_dim=12,
            encoder_hidden_dim=16,
            prediction_hidden_dim=24,
            joint_hidden_dim=32,
        )

        self.assertEqual(model.encoder_emb.weight.shape, (10, 8))
        self.assertEqual(model.pred_emb.weight.shape, (20, 12))
        self.assertEqual(model.encoder_lstm.hidden_size, 16)
        self.assertEqual(model.pred_lstm.hidden_size, 24)
        self.assertEqual(model.joint_fc1.in_features, 40)
        self.assertEqual(model.joint_fc1.out_features, 32)


if __name__ == "__main__":
    unittest.main()
