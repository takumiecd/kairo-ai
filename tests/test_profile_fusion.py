import unittest

import torch

from dataset.vocab import build_output_vocab
from decode.beam import beam_search_decode
from decode.profile_fusion import ProfileFusion
from decode.profile_fusion import build_word_bonus
from user_profile.builder import ProfileBuilder
from user_profile.schema import Profile


class TrieBuildAndPartialBonusTest(unittest.TestCase):
    def test_partial_bonus_accrues_along_matching_path(self):
        fusion = ProfileFusion.from_word_bonus({"東京": 1.0}, beta=0.1)
        state = fusion.initial_state()

        state, delta1 = fusion.step(state, "東")
        self.assertAlmostEqual(delta1, 0.1)  # depth 1, no word yet
        self.assertAlmostEqual(state.potential, 0.1)

        state, delta2 = fusion.step(state, "京")
        # depth 2 (0.2) + completed word bonus 1.0, minus the 0.1 already counted
        self.assertAlmostEqual(state.potential, 1.0 + 0.2)
        self.assertAlmostEqual(delta1 + delta2, state.potential)

    def test_mismatch_drops_partial_bonus_automatically(self):
        fusion = ProfileFusion.from_word_bonus({"東京": 1.0}, beta=0.1)
        state = fusion.initial_state()
        state, _ = fusion.step(state, "東")
        self.assertGreater(state.potential, 0.0)

        # "x" does not continue "東" in the trie -> partial credit vanishes,
        # and no new match starts either.
        state, delta = fusion.step(state, "x")
        self.assertEqual(state.potential, 0.0)
        self.assertEqual(delta, -0.1)

    def test_new_match_can_start_after_a_mismatch(self):
        fusion = ProfileFusion.from_word_bonus({"京都": 2.0}, beta=0.1)
        state = fusion.initial_state()
        state, _ = fusion.step(state, "東")  # no match, root has no "東" child... wait it does not exist
        self.assertEqual(state.potential, 0.0)
        state, _ = fusion.step(state, "京")
        self.assertAlmostEqual(state.potential, 0.1)
        state, _ = fusion.step(state, "都")
        self.assertAlmostEqual(state.potential, 2.0 + 0.2)


class ParallelPathTest(unittest.TestCase):
    """「東京」と「東京都」のような接頭辞関係にある語の並行処理。"""

    def test_shorter_word_completes_while_longer_continues(self):
        fusion = ProfileFusion.from_word_bonus({"東京": 1.0, "東京都": 3.0}, beta=0.1)
        state = fusion.initial_state()

        state, _ = fusion.step(state, "東")
        state, delta = fusion.step(state, "京")
        # "東京" completes here; the same node also has a child toward "都".
        self.assertAlmostEqual(state.total_word_bonus, 1.0)
        self.assertAlmostEqual(state.potential, 1.0 + 0.1 * 2)

        state, delta = fusion.step(state, "都")
        # "東京都" now also completes, bonus accumulates on top of "東京"'s.
        self.assertAlmostEqual(state.total_word_bonus, 1.0 + 3.0)
        self.assertAlmostEqual(state.potential, 4.0 + 0.1 * 3)

    def test_two_distinct_start_offsets_both_stay_active(self):
        # "京都" and "都" both valid; after consuming "京都", both a match
        # starting at position 0 ("京都") and a candidate starting at
        # position 1 ("都" alone, not present here but structurally must
        # not crash) are tracked independently.
        fusion = ProfileFusion.from_word_bonus({"京都": 2.0, "都": 0.5}, beta=0.1)
        state = fusion.initial_state()
        state, _ = fusion.step(state, "京")
        state, _ = fusion.step(state, "都")
        # "京都" (from offset 0) and "都" (from offset 1, started fresh at
        # this step) both complete simultaneously.
        self.assertAlmostEqual(state.total_word_bonus, 2.5)


class TelescopingTest(unittest.TestCase):
    def test_sum_of_deltas_equals_final_potential(self):
        fusion = ProfileFusion.from_word_bonus(
            {"東京": 1.0, "東京都": 3.0, "東京タワー": 2.0}, beta=0.05
        )
        state = fusion.initial_state()
        total_delta = 0.0
        for char in "東京タワー":
            state, delta = fusion.step(state, char)
            total_delta += delta
        self.assertAlmostEqual(total_delta, state.potential)

    def test_sum_of_deltas_equals_final_potential_with_mismatches(self):
        fusion = ProfileFusion.from_word_bonus({"猫": 1.0, "犬": 0.5}, beta=0.2)
        state = fusion.initial_state()
        total_delta = 0.0
        for char in "猫だと犬とネコ":
            state, delta = fusion.step(state, char)
            total_delta += delta
        self.assertAlmostEqual(total_delta, state.potential)


class BuildWordBonusTest(unittest.TestCase):
    def test_combines_explicit_and_implicit_with_lambda_weights(self):
        profile = Profile()
        builder = ProfileBuilder(profile=profile)
        builder.apply_correction("けいやく", "契約")
        builder.apply_commit("契約")

        word_bonus = build_word_bonus(profile, lambda_exp=2.0, lambda_imp=1.0)
        self.assertIn("契約", word_bonus)
        # both explicit (accept_count=1) and implicit (count=1) contribute.
        self.assertGreater(word_bonus["契約"], 0.0)

    def test_implicit_top_k_filters_out_low_ranked_words(self):
        profile = Profile()
        builder = ProfileBuilder(profile=profile)
        for _ in range(10):
            builder.apply_commit("よく")
        builder.apply_commit("たまにしか")

        word_bonus_unfiltered = build_word_bonus(profile, implicit_top_k=10_000)
        self.assertIn("たまにしか", word_bonus_unfiltered)

        # "たまにしか" has a much lower decayed count than "よく" (1 touch vs
        # 10); with top_k=1 only the single highest-ranked implicit word
        # ("よく") should survive.
        word_bonus_top1 = build_word_bonus(profile, implicit_top_k=1)
        self.assertNotIn("たまにしか", word_bonus_top1)
        self.assertIn("よく", word_bonus_top1)


class DummyBeamModel:
    """`decode/tests` の DummyBeamModel と同じ形。先頭2文字の分岐を作る。"""

    def __init__(self, first_id: int, second_id: int, blank_id: int) -> None:
        self.first_id = first_id
        self.second_id = second_id
        self.blank_id = blank_id

    def __call__(self, x, y):
        vocab_size = max(self.first_id, self.second_id, self.blank_id) + 1
        logits = torch.full((1, x.shape[1], y.shape[1], vocab_size), -100.0)
        emitted_count = y.shape[1] - 1
        if emitted_count == 0:
            # second_id is slightly favored by the raw model, so that a
            # profile bonus on first_id is needed to flip the ranking.
            logits[0, :, y.shape[1] - 1, self.first_id] = 7.0
            logits[0, :, y.shape[1] - 1, self.second_id] = 7.5
            logits[0, :, y.shape[1] - 1, self.blank_id] = 0.0
        else:
            logits[0, :, y.shape[1] - 1, self.blank_id] = 8.0
        return logits


class BeamSearchNoProfileMatchesBaselineTest(unittest.TestCase):
    def test_profile_none_matches_existing_behavior(self):
        output_vocab = build_output_vocab(["した"])
        first_id = output_vocab.token_to_id["し"]
        second_id = output_vocab.token_to_id["た"]
        blank_id = output_vocab.token_to_id["<blank>"]
        model = DummyBeamModel(first_id, second_id, blank_id)

        candidates = beam_search_decode(
            model,
            input_ids=[2],
            output_vocab=output_vocab,
            beam_width=2,
            expansion_width=2,
            max_symbols_per_step=2,
        )
        candidates_explicit_none = beam_search_decode(
            model,
            input_ids=[2],
            output_vocab=output_vocab,
            beam_width=2,
            expansion_width=2,
            max_symbols_per_step=2,
            profile=None,
        )
        self.assertEqual(candidates, candidates_explicit_none)


class BeamSearchProfileIntegrationTest(unittest.TestCase):
    def test_profile_bonus_promotes_registered_candidate_rank(self):
        output_vocab = build_output_vocab(["した"])
        first_id = output_vocab.token_to_id["し"]  # underdog per raw model
        second_id = output_vocab.token_to_id["た"]  # favored per raw model
        blank_id = output_vocab.token_to_id["<blank>"]
        model = DummyBeamModel(first_id, second_id, blank_id)

        baseline = beam_search_decode(
            model,
            input_ids=[2],
            output_vocab=output_vocab,
            beam_width=2,
            expansion_width=2,
            max_symbols_per_step=2,
        )
        self.assertEqual(baseline[0].text, "た")

        profile = Profile()
        builder = ProfileBuilder(profile=profile)
        builder.apply_correction("input", "し")
        builder.apply_correction("input", "し")
        builder.apply_correction("input", "し")

        with_profile = beam_search_decode(
            model,
            input_ids=[2],
            output_vocab=output_vocab,
            beam_width=2,
            expansion_width=2,
            max_symbols_per_step=2,
            profile=profile,
            profile_fusion_weight=2.0,
        )
        self.assertEqual(with_profile[0].text, "し")


if __name__ == "__main__":
    unittest.main()
