import json
import tempfile
import unittest
from pathlib import Path

from dataset.split import load_jsonl
from dataset.split import split_records
from dataset.split import write_jsonl


class DatasetSplitTest(unittest.TestCase):
    def test_split_records_uses_requested_sizes(self):
        records = [{"id": index} for index in range(10)]

        train, valid, test = split_records(
            records,
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1,
            seed=0,
        )

        self.assertEqual(len(train), 8)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(test), 1)

    def test_split_records_is_deterministic(self):
        records = [{"id": index} for index in range(10)]

        first = split_records(records, 0.8, 0.1, 0.1, seed=42)
        second = split_records(records, 0.8, 0.1, 0.1, seed=42)

        self.assertEqual(first, second)

    def test_split_records_rejects_bad_ratios(self):
        with self.assertRaises(ValueError):
            split_records([{"id": 1}], 0.8, 0.1, 0.2, seed=0)

    def test_jsonl_round_trip(self):
        records = [{"text": "修正した"}, {"text": "追加した"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"

            write_jsonl(path, records)
            loaded = load_jsonl(path)

        self.assertEqual(loaded, records)

    def test_written_jsonl_is_utf8_json_per_line(self):
        records = [{"text": "修正した"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"

            write_jsonl(path, records)
            line = path.read_text(encoding="utf-8").strip()

        self.assertEqual(json.loads(line), records[0])


if __name__ == "__main__":
    unittest.main()
