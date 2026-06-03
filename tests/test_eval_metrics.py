import unittest

from eval.metrics import cer
from eval.metrics import cer_result
from eval.metrics import edit_distance
from eval.metrics import mean_cer


class EvalMetricsTest(unittest.TestCase):
    def test_edit_distance_exact_match(self):
        self.assertEqual(edit_distance("修正した", "修正した"), 0)

    def test_edit_distance_substitution(self):
        self.assertEqual(edit_distance("修整した", "修正した"), 1)

    def test_edit_distance_insert_delete(self):
        self.assertEqual(edit_distance("修正しました", "修正した"), 2)

    def test_cer(self):
        self.assertAlmostEqual(cer("修整した", "修正した"), 0.25)

    def test_cer_result_keeps_counts(self):
        result = cer_result("abc", "abcd")

        self.assertEqual(result.edits, 1)
        self.assertEqual(result.reference_length, 4)
        self.assertAlmostEqual(result.value, 0.25)

    def test_mean_cer_uses_corpus_level_denominator(self):
        self.assertAlmostEqual(
            mean_cer(["修整した", "abc"], ["修正した", "abcd"]),
            2 / 8,
        )

    def test_mean_cer_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            mean_cer(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
