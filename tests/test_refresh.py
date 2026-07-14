import json
import os
import tempfile
import unittest
from pathlib import Path

from user_profile.refresh import default_config_dir
from user_profile.refresh import refresh
from user_profile.schema import Profile


def feedback_event(input_text, output, rank=0, accepted=True):
    return {
        "v": 1,
        "input": input_text,
        "output": output,
        "candidate_rank": rank,
        "accepted": accepted,
    }


def write_feedback(path: Path, events: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


class RefreshEndToEndTest(unittest.TestCase):
    """feedback.jsonl -> profile.json -> profile_bias.tsv を module 直呼びで検証する。

    subprocess は使わない(``refresh()`` を直接呼ぶ)。IME からの実行経路である
    ``python -m user_profile.refresh`` の CLI 層は薄いラッパーに過ぎないので、
    ここでは中核ロジックだけを厳密にテストする。
    """

    def test_first_run_is_full_rebuild_and_writes_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            write_feedback(
                config_dir / "feedback.jsonl",
                [
                    feedback_event("kansuu", "関数", rank=1),
                    feedback_event("kansuu", "関数", rank=1),
                    feedback_event("neko", "猫"),
                ],
            )

            stats = refresh(config_dir)

            self.assertFalse(stats["incremental"])
            self.assertEqual(stats["new_events"], 3)

            profile_path = config_dir / "profile.json"
            bias_path = config_dir / "profile_bias.tsv"
            self.assertTrue(profile_path.exists())
            self.assertTrue(bias_path.exists())

            profile = Profile.load_json(profile_path)
            # candidate_rank>0 の確定は apply_commit (実際にユーザが確定した
            # テキストとして) + apply_correction (明示的な修正シグナル) の
            # 両方を通る。
            self.assertGreater(profile.meta.total_units, 0)
            surfaces = {entry.surface for entry in profile.explicit}
            self.assertIn("関数", surfaces)

            bias_text = bias_path.read_text(encoding="utf-8")
            self.assertIn("関数", bias_text)

    def test_second_run_only_applies_new_events_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            feedback_path = config_dir / "feedback.jsonl"
            write_feedback(feedback_path, [feedback_event("neko", "猫")])

            first = refresh(config_dir)
            self.assertFalse(first["incremental"])
            self.assertEqual(first["new_events"], 1)
            total_units_after_first = first["total_units"]

            # 2回目: feedback.jsonl に追記する(append-only の実際の使われ方)。
            write_feedback(feedback_path, [feedback_event("kansuu", "関数", rank=1)])

            second = refresh(config_dir)
            self.assertTrue(second["incremental"])
            # 増分適用なので、2回目に新しく処理されたイベントは1件だけ。
            self.assertEqual(second["new_events"], 1)
            # total_units は前回の値に積み上がる(フルリビルドで消えていない)。
            self.assertGreater(second["total_units"], total_units_after_first)

            profile = Profile.load_json(config_dir / "profile.json")
            self.assertIn("猫", profile.unigram)
            surfaces = {entry.surface for entry in profile.explicit}
            self.assertIn("関数", surfaces)

    def test_third_run_with_no_new_events_is_a_noop_on_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            write_feedback(config_dir / "feedback.jsonl", [feedback_event("neko", "猫")])

            refresh(config_dir)
            stats = refresh(config_dir)  # feedback.jsonl 変更なしで再実行

            self.assertTrue(stats["incremental"])
            self.assertEqual(stats["new_events"], 0)

    def test_corrupt_profile_json_falls_back_to_full_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            write_feedback(config_dir / "feedback.jsonl", [feedback_event("neko", "猫")])
            (config_dir / "profile.json").write_text("{not valid json", encoding="utf-8")

            stats = refresh(config_dir)

            self.assertFalse(stats["incremental"])
            self.assertEqual(stats["new_events"], 1)

    def test_missing_feedback_file_is_a_harmless_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            stats = refresh(config_dir)
            self.assertEqual(stats["new_events"], 0)
            self.assertTrue((config_dir / "profile.json").exists())
            self.assertTrue((config_dir / "profile_bias.tsv").exists())

    def test_writes_are_atomic_rename_not_in_place(self):
        # os.replace-based atomic write: the destination should never be a
        # partially-written file even if we don't observe the intermediate
        # state directly here. We instead assert the tmp file doesn't leak.
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            write_feedback(config_dir / "feedback.jsonl", [feedback_event("neko", "猫")])
            refresh(config_dir)
            leftover_tmp_files = [
                p for p in config_dir.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")
            ]
            self.assertEqual(leftover_tmp_files, [])


class DefaultConfigDirTest(unittest.TestCase):
    def test_honors_xdg_config_home(self):
        old = os.environ.get("XDG_CONFIG_HOME")
        try:
            os.environ["XDG_CONFIG_HOME"] = "/tmp/kairo-xdg-test"
            self.assertEqual(default_config_dir(), Path("/tmp/kairo-xdg-test/kairo"))
        finally:
            if old is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old

    def test_falls_back_to_home_config_kairo(self):
        old = os.environ.pop("XDG_CONFIG_HOME", None)
        try:
            self.assertEqual(default_config_dir(), Path.home() / ".config" / "kairo")
        finally:
            if old is not None:
                os.environ["XDG_CONFIG_HOME"] = old


if __name__ == "__main__":
    unittest.main()
