import unittest

from model.edit_transducer import KairoEditTransducer
from train.common.data import build_vocabs_from_records
from train.edit.data import collate_edit_batch
from train.edit.data import encode_edit_example


class KairoEditTransducerTest(unittest.TestCase):
    def test_forward_returns_operation_and_insert_logits(self):
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

        op_logits, insert_logits = model(
            batch["inputs"],
            batch["previous_tokens"],
            batch["action_input_ops"],
            batch["action_input_insert_tokens"],
        )

        self.assertEqual(op_logits.shape[:2], batch["action_target_ops"].shape)
        self.assertEqual(op_logits.shape[-1], 4)
        self.assertEqual(insert_logits.shape[-1], len(vocabs.output_vocab.id_to_token))

    def test_forward_accepts_empty_previous_hypothesis(self):
        records = [{"input": "kyouhahashide", "target": "今日は箸で"}]
        vocabs = build_vocabs_from_records(records)
        example = encode_edit_example("kyouhahashide", "", "今日は箸で", vocabs)
        batch = collate_edit_batch([example], vocabs)
        model = KairoEditTransducer(
            input_vocab_size=len(vocabs.input_vocab.id_to_token),
            output_vocab_size=len(vocabs.output_vocab.id_to_token),
            input_embed_dim=8,
            output_embed_dim=8,
            action_embed_dim=4,
            hidden_dim=16,
        )

        op_logits, insert_logits = model(
            batch["inputs"],
            batch["previous_tokens"],
            batch["action_input_ops"],
            batch["action_input_insert_tokens"],
        )

        self.assertEqual(op_logits.shape[1], batch["action_target_ops"].shape[1])
        self.assertEqual(insert_logits.shape[-1], len(vocabs.output_vocab.id_to_token))


if __name__ == "__main__":
    unittest.main()
