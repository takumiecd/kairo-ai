import unittest
from dataclasses import dataclass

import torch

from bench.rnnt import benchmark_prefix_decode
from bench.rnnt import summarize_seconds
from bench.rnnt import percentile


@dataclass(frozen=True)
class DummyExample:
    input_text: str


class RnntBenchmarkTest(unittest.TestCase):
    def test_percentile_interpolates_between_points(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertEqual(percentile([10.0, 20.0], 95), 19.5)

    def test_summarize_seconds_reports_milliseconds(self):
        summary = summarize_seconds([0.001, 0.002, 0.004])

        self.assertEqual(summary.count, 3)
        self.assertAlmostEqual(summary.mean_ms, 7.0 / 3.0)
        self.assertEqual(summary.min_ms, 1.0)
        self.assertEqual(summary.max_ms, 4.0)

    def test_empty_percentile_rejected(self):
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_prefix_decode_measures_each_prefix(self):
        result = benchmark_prefix_decode(
            decoder="greedy",
            decode_fn=lambda ids: "x" * len(ids),
            encode_input=lambda text: list(range(len(text))),
            examples=[DummyExample("abcd")],
            warmup_examples=0,
            max_examples=1,
            prefix_stride=2,
            min_prefix_chars=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.decoder, "greedy_prefix")
        self.assertEqual(result.samples, 3)
        self.assertEqual(result.input_chars, 8)


if __name__ == "__main__":
    unittest.main()
