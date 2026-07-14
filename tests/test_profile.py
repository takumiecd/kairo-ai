import json
import tempfile
import unittest
from pathlib import Path

from user_profile.builder import ProfileBuilder
from user_profile.builder import split_words
from user_profile.from_corpus import stream_snapshots
from user_profile.from_feedback import apply_events
from user_profile.from_feedback import build_profile_from_feedback
from user_profile.schema import Profile


def feedback_event(input_text, output, rank=0, accepted=True):
    return {
        "v": 1,
        "input": input_text,
        "output": output,
        "candidate_rank": rank,
        "accepted": accepted,
    }


class SplitWordsTest(unittest.TestCase):
    def test_splits_on_script_and_ascii_boundaries(self):
        self.assertEqual(split_words("東京タワーに行く"), ["東京", "タワー", "に", "行", "く"])
        self.assertEqual(split_words("hello world"), ["hello", "world"])
        self.assertEqual(split_words("import numpy as np"), ["import", "numpy", "as", "np"])

    def test_punctuation_is_a_boundary_not_a_token(self):
        self.assertEqual(split_words("猫。犬！"), ["猫", "犬"])
        self.assertEqual(split_words(""), [])


class ApplyCommitTest(unittest.TestCase):
    def test_updates_unigram_recency_and_total_units(self):
        builder = ProfileBuilder()
        builder.apply_commit("ねこ", reading="neko")
        self.assertEqual(builder.profile.meta.total_units, 2)
        self.assertIn("ねこ", builder.profile.unigram)
        entry = builder.profile.unigram["ねこ"]
        self.assertEqual(entry.count, 1.0)
        self.assertEqual(entry.reading, "neko")
        self.assertEqual(entry.last_used, 2)
        self.assertEqual(builder.profile.recency, ["ねこ"])

    def test_repeated_commit_increments_count(self):
        # Use a very long half-life so decay between the two touches is negligible
        # and we're really testing the +1 increment, not the decay math.
        builder = ProfileBuilder(half_life=10**12)
        builder.apply_commit("ねこ")
        builder.apply_commit("ねこ")
        self.assertAlmostEqual(builder.profile.unigram["ねこ"].count, 2.0, places=6)

    def test_domain_vector_normalizes_towards_hint(self):
        builder = ProfileBuilder()
        for _ in range(5):
            builder.apply_commit("def foo(): pass", domain="code")
        self.assertAlmostEqual(builder.profile.domain["code"], 1.0, places=6)
        self.assertAlmostEqual(
            sum(builder.profile.domain.values()), 1.0, places=6
        )

    def test_lang_ja_ratio_tracks_japanese_char_share(self):
        builder = ProfileBuilder()
        builder.apply_commit("猫犬")  # all-Japanese, 2 chars
        self.assertAlmostEqual(builder.profile.lang["ja_ratio"], 1.0, places=6)
        builder.apply_commit("ab")  # all-ASCII, 2 chars
        self.assertAlmostEqual(builder.profile.lang["ja_ratio"], 0.5, places=6)

    def test_empty_commit_is_noop(self):
        builder = ProfileBuilder()
        builder.apply_commit("")
        self.assertEqual(builder.profile.meta.total_units, 0)
        self.assertEqual(builder.profile.unigram, {})


class ApplyCorrectionRejectionTest(unittest.TestCase):
    def test_correction_creates_and_increments_explicit_entry(self):
        builder = ProfileBuilder()
        builder.apply_correction("wagahai", "吾輩")
        builder.apply_correction("wagahai", "吾輩")
        self.assertEqual(len(builder.profile.explicit), 1)
        entry = builder.profile.explicit[0]
        self.assertEqual(entry.accept_count, 2)
        self.assertEqual(entry.reject_count, 0)
        self.assertEqual(entry.source, "correction")

    def test_rejection_creates_and_increments_reject_count(self):
        builder = ProfileBuilder()
        builder.apply_rejection("neko", "猫")
        self.assertEqual(len(builder.profile.explicit), 1)
        entry = builder.profile.explicit[0]
        self.assertEqual(entry.reject_count, 1)
        self.assertEqual(entry.accept_count, 0)

    def test_correction_and_rejection_share_the_same_entry(self):
        builder = ProfileBuilder()
        builder.apply_correction("neko", "猫")
        builder.apply_rejection("neko", "猫")
        self.assertEqual(len(builder.profile.explicit), 1)
        entry = builder.profile.explicit[0]
        self.assertEqual(entry.accept_count, 1)
        self.assertEqual(entry.reject_count, 1)

    def test_explicit_does_not_decay(self):
        builder = ProfileBuilder(half_life=1)
        builder.apply_correction("neko", "猫")
        for _ in range(10):
            builder.apply_commit("x" * 100)
        self.assertEqual(builder.profile.explicit[0].accept_count, 1)


class LazyDecayTest(unittest.TestCase):
    def test_decayed_count_halves_after_one_half_life(self):
        builder = ProfileBuilder(half_life=100)
        builder.apply_commit("ねこ")  # N=2, last_used=2, count=1
        builder.apply_commit("x" * 98)  # N=100, unrelated commit advances N only
        # elapsed = 100 - 2 = 98, not yet a full half-life.
        self.assertGreater(builder.decayed_count("ねこ"), 0.5)
        builder.apply_commit("y" * 100)  # N=200 -> elapsed=198
        # roughly two half-lives since last touch of "ねこ" (elapsed=198 ~ 1.98H)
        expected = 1.0 * 2.0 ** (-198 / 100)
        self.assertAlmostEqual(builder.decayed_count("ねこ"), expected, places=9)

    def test_decayed_count_zero_for_unknown_word(self):
        builder = ProfileBuilder()
        self.assertEqual(builder.decayed_count("nope"), 0.0)

    def test_touching_a_word_again_resets_the_decay_clock(self):
        builder = ProfileBuilder(half_life=10)
        builder.apply_commit("ab")  # touches "ab" at N=2
        builder.apply_commit("cd" * 20)  # advance N a lot, no touch of "ab"
        decayed_before = builder.decayed_count("ab")
        builder.apply_commit("ab")  # touch again -> count bumps and clock resets
        self.assertGreater(builder.decayed_count("ab"), decayed_before)


class UnigramCapTest(unittest.TestCase):
    def test_cap_evicts_smallest_decayed_count_first(self):
        builder = ProfileBuilder(unigram_cap=3, half_life=1_000_000)
        for word in ["a", "b", "c"]:
            builder.apply_commit(word)
        # "a" is oldest / least-recently touched -> should be evicted first.
        builder.apply_commit("d")
        self.assertEqual(len(builder.profile.unigram), 3)
        self.assertNotIn("a", builder.profile.unigram)
        self.assertIn("d", builder.profile.unigram)

    def test_never_exceeds_cap(self):
        builder = ProfileBuilder(unigram_cap=5)
        for index in range(50):
            builder.apply_commit(f"w{index}")
        self.assertLessEqual(len(builder.profile.unigram), 5)


class SerializationTest(unittest.TestCase):
    def test_roundtrip_via_dict(self):
        builder = ProfileBuilder()
        builder.apply_commit("東京タワー", reading="toukyoutawaa")
        builder.apply_correction("neko", "猫")
        data = builder.profile.to_dict()
        restored = Profile.from_dict(data)
        self.assertEqual(restored.to_dict(), data)
        self.assertEqual(restored.meta.total_units, builder.profile.meta.total_units)
        self.assertEqual(restored.unigram.keys(), builder.profile.unigram.keys())

    def test_roundtrip_via_json_file(self):
        builder = ProfileBuilder()
        builder.apply_commit("猫が鳴く")
        builder.apply_rejection("neko", "猫")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            builder.profile.save_json(path)
            self.assertTrue(path.exists())
            loaded = Profile.load_json(path)
        self.assertEqual(loaded.to_dict(), builder.profile.to_dict())

    def test_recency_and_domain_survive_roundtrip(self):
        builder = ProfileBuilder(recency_size=2)
        builder.apply_commit("a")
        builder.apply_commit("b", domain="code")
        builder.apply_commit("c", domain="code")
        restored = Profile.from_dict(builder.profile.to_dict())
        self.assertEqual(restored.recency, builder.profile.recency)
        self.assertEqual(restored.domain, builder.profile.domain)


class FromCorpusSnapshotTest(unittest.TestCase):
    def test_snapshots_pair_previous_state_with_next_sentence(self):
        sentences = ["猫が歩く。", "犬が走る。", "鳥が飛ぶ。"]
        pairs = list(stream_snapshots(sentences, snapshot_every=1))
        self.assertEqual(len(pairs), 3)
        first_snapshot, first_next = pairs[0]
        self.assertEqual(first_next, "猫が歩く。")
        # u_0 is empty: nothing committed yet before the first sentence.
        self.assertEqual(first_snapshot["meta"]["total_units"], 0)
        self.assertEqual(first_snapshot["implicit"]["unigram"], {})

        second_snapshot, second_next = pairs[1]
        self.assertEqual(second_next, "犬が走る。")
        self.assertEqual(second_snapshot["meta"]["total_units"], len(sentences[0]))

    def test_snapshots_grow_monotonically(self):
        sentences = [f"文{index}が続く。" for index in range(20)]
        pairs = list(stream_snapshots(sentences, snapshot_every=1))
        totals = [snapshot["meta"]["total_units"] for snapshot, _ in pairs]
        self.assertEqual(totals, sorted(totals))
        self.assertTrue(all(earlier <= later for earlier, later in zip(totals, totals[1:])))
        # Early snapshots must be sparse (cold start), matching the corpus stream.
        self.assertEqual(totals[0], 0)
        self.assertGreater(totals[-1], totals[0])

    def test_snapshot_every_controls_sampling_rate(self):
        sentences = ["a", "b", "c", "d"]
        pairs = list(stream_snapshots(sentences, snapshot_every=2))
        self.assertEqual([text for _, text in pairs], ["a", "c"])

    def test_dict_items_carry_domain_and_reading(self):
        sentences = [{"text": "コードを書く", "domain": "code", "reading": "code_wo_kaku"}]
        pairs = list(stream_snapshots(sentences))
        self.assertEqual(len(pairs), 1)

    def test_synthetic_explicit_rate_injects_future_words(self):
        sentences = ["猫が歩く。", "犬が走る。"] * 5
        builder = ProfileBuilder()
        list(
            stream_snapshots(
                sentences,
                builder=builder,
                synthetic_explicit_rate=1.0,  # force injection deterministically
                lookahead=10,
                seed=0,
            )
        )
        self.assertGreater(len(builder.profile.explicit), 0)

    def test_zero_synthetic_rate_injects_nothing(self):
        sentences = ["猫が歩く。", "犬が走る。"]
        builder = ProfileBuilder()
        list(stream_snapshots(sentences, builder=builder, synthetic_explicit_rate=0.0))
        self.assertEqual(builder.profile.explicit, [])


class FromFeedbackEquivalenceTest(unittest.TestCase):
    def _events(self):
        return [
            feedback_event("neko", "猫"),
            feedback_event("neko", "猫", rank=2),
            feedback_event("inu", "犬", accepted=False),
            feedback_event("tori", "鳥"),
        ]

    def test_from_feedback_matches_direct_builder_calls(self):
        events = self._events()

        builder_via_feedback = ProfileBuilder()
        apply_events(builder_via_feedback, events)

        builder_direct = ProfileBuilder()
        builder_direct.apply_commit("猫", reading="neko")
        builder_direct.apply_commit("猫", reading="neko")
        builder_direct.apply_correction("neko", "猫")
        builder_direct.apply_rejection("inu", "犬")
        builder_direct.apply_commit("鳥", reading="tori")

        self.assertEqual(
            builder_via_feedback.profile.to_dict(), builder_direct.profile.to_dict()
        )

    def test_build_profile_from_feedback_reads_jsonl_file(self):
        events = self._events()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.jsonl"
            with path.open("w", encoding="utf-8") as file:
                for event in events:
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")
            profile = build_profile_from_feedback([path])
        self.assertGreater(profile.meta.total_units, 0)
        self.assertEqual(len(profile.explicit), 2)  # "neko/猫" correction + "inu/犬" rejection


if __name__ == "__main__":
    unittest.main()
