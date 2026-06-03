import unittest

from train.data import build_vocabs_from_records
from train.edit_data import ACTION_BOS
from train.edit_data import DELETE
from train.edit_data import INSERT
from train.edit_data import KEEP
from train.edit_data import STOP
from train.edit_data import apply_edit_script
from train.edit_data import build_min_edit_script
from train.edit_data import collate_edit_batch
from train.edit_data import encode_edit_example


class EditDataTest(unittest.TestCase):
    def test_min_edit_script_replaces_token_with_delete_insert(self):
        records = [{"input": "kyouhahashide", "target": "今日は箸で"}]
        vocabs = build_vocabs_from_records(records)
        previous_ids = vocabs.output_vocab.encode("今日は橋で")
        target_ids = vocabs.output_vocab.encode("今日は箸で")

        actions = build_min_edit_script(previous_ids, target_ids)
        op_ids = [action.op_id for action in actions]

        self.assertIn(DELETE, op_ids)
        self.assertIn(INSERT, op_ids)
        self.assertEqual(actions[-1].op_id, STOP)
        self.assertEqual(apply_edit_script(previous_ids, actions), target_ids)

    def test_collate_edit_batch_builds_teacher_forcing_inputs(self):
        records = [{"input": "kyouhahashide", "target": "今日は箸で"}]
        vocabs = build_vocabs_from_records(records)
        example = encode_edit_example("kyouhahashide", "今日は橋で", "今日は箸で", vocabs)

        batch = collate_edit_batch([example], vocabs)

        self.assertEqual(batch["inputs"].shape[0], 1)
        self.assertEqual(batch["previous_tokens"].shape[0], 1)
        self.assertEqual(batch["action_input_ops"][0, 0].item(), ACTION_BOS)
        self.assertEqual(batch["action_target_ops"][0, 0].item(), KEEP)
        self.assertEqual(batch["action_target_ops"][0, -1].item(), STOP)


if __name__ == "__main__":
    unittest.main()
