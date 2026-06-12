import unittest
from dataclasses import dataclass

import torch

from bench.rnnt import benchmark_prefix_decode
from bench.rnnt import filter_dataset
from bench.rnnt import summarize_seconds
from bench.rnnt import summarize_dataset
from bench.rnnt import percentile
from train.rnnt.data import EncodedExample
from train.rnnt.data import JsonlTransducerDataset


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

    def test_dataset_filter_uses_lattice_cells(self):
        dataset = JsonlTransducerDataset(
            [
                EncodedExample([1, 2], [3], "ab", "x"),
                EncodedExample([1, 2, 3, 4], [5, 6, 7], "abcd", "xyz"),
            ]
        )

        filtered = filter_dataset(
            dataset,
            max_input_len=None,
            max_target_len=None,
            max_lattice_cells=5,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.examples[0].input_text, "ab")

    def test_dataset_summary_reports_lengths(self):
        dataset = JsonlTransducerDataset(
            [
                EncodedExample([1, 2], [3], "ab", "x"),
                EncodedExample([1, 2, 3, 4], [5, 6], "abcd", "xy"),
            ]
        )

        summary = summarize_dataset(dataset)

        self.assertEqual(summary.examples, 2)
        self.assertEqual(summary.input_len.max, 4)
        self.assertEqual(summary.target_len.max, 2)
        self.assertEqual(summary.lattice_cells.max, 12)


if __name__ == "__main__":
    unittest.main()
