import tempfile
import unittest
from pathlib import Path

from dataset.split import write_jsonl
from eval.run_test import load_records


class EvalRunTestTest(unittest.TestCase):
    def test_load_records_filters_by_max_len_before_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            write_jsonl(
                path,
                [
                    {"input": "a" * 20, "target": "長い"},
                    {"input": "abc", "target": "短い"},
                    {"input": "def", "target": "短い2"},
                ],
            )

            records, skipped = load_records(path, limit=2, max_len=10)

        self.assertEqual(len(records), 2)
        self.assertEqual(skipped, 1)
        self.assertEqual(records[0]["input"], "abc")


if __name__ == "__main__":
    unittest.main()
