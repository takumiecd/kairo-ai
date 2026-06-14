import json
import tempfile
import unittest
from pathlib import Path

from dataset.source_feedback import aggregate
from dataset.source_feedback import build_records
from dataset.source_feedback import load_feedback_events


def event(input_text, output, rank=0, accepted=True):
    return {
        "v": 1,
        "input": input_text,
        "output": output,
        "candidate_rank": rank,
        "accepted": accepted,
    }


class SourceFeedbackTest(unittest.TestCase):
    def test_load_skips_blank_and_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.jsonl"
            path.write_text(
                json.dumps(event("neko", "猫")) + "\n\n" + "{not json}\n",
                encoding="utf-8",
            )
            events = load_feedback_events([path])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["output"], "猫")

    def test_aggregate_counts_and_corrections(self):
        events = [
            event("neko", "猫"),
            event("neko", "猫", rank=2),
            event("inu", "犬", accepted=False),
            event("", "空"),
        ]
        stats = aggregate(events)
        self.assertEqual(stats[("neko", "猫")].count, 2)
        self.assertEqual(stats[("neko", "猫")].corrections, 1)
        self.assertNotIn(("inu", "犬"), stats)  # not accepted
        self.assertNotIn(("", "空"), stats)  # empty input

    def test_build_records_min_count_and_correction_repeat(self):
        events = [
            event("neko", "猫"),
            event("aaa", "あ", rank=1),
        ]
        stats = aggregate(events)
        records = build_records(stats, min_count=2, repeat_corrections=3)
        # neko/猫 count=1 dropped; aaa/あ is a correction -> repeated 3x.
        self.assertEqual(records, [{"input": "aaa", "target": "あ"}] * 3)

    def test_records_use_input_target_keys(self):
        stats = aggregate([event("neko", "猫")])
        records = build_records(stats, min_count=1, repeat_corrections=1)
        self.assertEqual(records, [{"input": "neko", "target": "猫"}])


if __name__ == "__main__":
    unittest.main()
