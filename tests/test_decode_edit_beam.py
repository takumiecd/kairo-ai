import unittest

import torch

from dataset.vocab import build_input_vocab
from dataset.vocab import build_output_vocab
from decode.edit_beam import edit_beam_search_decode
from train.edit_data import ACTION_BOS
from train.edit_data import DELETE
from train.edit_data import INSERT
from train.edit_data import KEEP
from train.edit_data import STOP


class DummyEditModel:
    def __init__(self, output_vocab, bridge_id: int, chopstick_id: int) -> None:
        self.output_vocab = output_vocab
        self.bridge_id = bridge_id
        self.chopstick_id = chopstick_id

    def predict_next(self, inputs, previous_tokens, action_context_ops, action_context_insert_tokens):
        op_logits = torch.full((1, 4), -100.0)
        insert_logits = torch.full((1, len(self.output_vocab.id_to_token)), -100.0)
        ops = action_context_ops[0].tolist()[1:]
        cursor = sum(1 for op in ops if op in (KEEP, DELETE))
        last_op = action_context_ops[0, -1].item()

        if last_op == DELETE:
            op_logits[0, INSERT] = 8.0
            op_logits[0, KEEP] = 0.0
            insert_logits[0, self.chopstick_id] = 8.0
            insert_logits[0, self.bridge_id] = 7.0
        elif cursor < previous_tokens.shape[1] and previous_tokens[0, cursor].item() == self.bridge_id:
            op_logits[0, DELETE] = 8.0
            op_logits[0, KEEP] = 0.0
        elif cursor >= previous_tokens.shape[1]:
            if last_op in (ACTION_BOS, KEEP, INSERT, DELETE):
                op_logits[0, STOP] = 8.0
        else:
            op_logits[0, KEEP] = 8.0
        return op_logits, insert_logits


class DecodeEditBeamTest(unittest.TestCase):
    def test_edit_beam_returns_revision_candidates(self):
        input_vocab = build_input_vocab(["kyouhahashide"])
        output_vocab = build_output_vocab(["今日は橋で", "今日は箸で"])
        bridge_id = output_vocab.token_to_id["橋"]
        chopstick_id = output_vocab.token_to_id["箸"]
        model = DummyEditModel(output_vocab, bridge_id, chopstick_id)

        candidates = edit_beam_search_decode(
            model,
            input_ids=input_vocab.encode("kyouhahashide"),
            previous_ids=output_vocab.encode("今日は橋で"),
            output_vocab=output_vocab,
            beam_width=2,
            expansion_width=2,
        )

        self.assertEqual(candidates[0].text, "今日は箸で")
        self.assertIn("今日は橋で", [candidate.text for candidate in candidates])
        self.assertAlmostEqual(sum(candidate.confidence for candidate in candidates), 1.0)


if __name__ == "__main__":
    unittest.main()
