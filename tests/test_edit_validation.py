import unittest

import torch

from dataset.vocab import build_input_vocab
from dataset.vocab import build_output_vocab
from train.data import TrainingVocabs
from train.edit_data import JsonlEditDataset
from train.edit_data import encode_edit_example
from train.edit_validation import evaluate_edit_decode_cer


class DummyKeepModel:
    def eval(self):
        return self

    def train(self):
        return self

    def predict_next(self, inputs, previous_tokens, action_context_ops, action_context_insert_tokens):
        op_logits = torch.full((1, 4), -100.0)
        ops = action_context_ops[0].tolist()[1:]
        cursor = sum(1 for op in ops if op in (0, 1))
        op_logits[0, 3 if cursor >= previous_tokens.shape[1] else 0] = 8.0
        insert_logits = torch.zeros((1, int(previous_tokens.max().item()) + 1))
        return op_logits, insert_logits


class EditValidationTest(unittest.TestCase):
    def test_greedy_validation_scores_edit_candidate(self):
        input_vocab = build_input_vocab(["abc"])
        output_vocab = build_output_vocab(["修正"])
        vocabs = TrainingVocabs(input_vocab=input_vocab, output_vocab=output_vocab)
        example = encode_edit_example("abc", "修正", "修正", vocabs)
        dataset = JsonlEditDataset([example])

        value = evaluate_edit_decode_cer(
            DummyKeepModel(),
            dataset,
            output_vocab,
            decoder="greedy",
            max_samples=1,
        )

        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
