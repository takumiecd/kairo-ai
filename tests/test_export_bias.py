import math
import unittest
from pathlib import Path

from user_profile.builder import DEFAULT_HALF_LIFE
from user_profile.export_bias import HEADER
from user_profile.export_bias import compute_explicit_bonus
from user_profile.export_bias import compute_implicit_bonus
from user_profile.export_bias import export_bias
from user_profile.export_bias import format_tsv
from user_profile.builder import ProfileBuilder
from user_profile.schema import ExplicitEntry
from user_profile.schema import Profile
from user_profile.schema import UnigramEntry


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _golden_profile() -> Profile:
    """A small, hand-checkable Profile.

    - explicit: one entry ("kansuu" -> "関数", accept=3, reject=1) so
      B_exp = log(1+3) - 1.0*log(1+1) = log(4) - log(2) = log(2).
    - unigram: "猫" with count=5.0 and last_used == total_units, so decay is
      exactly a no-op (elapsed=0) regardless of half_life -> decayed_count=5.0,
      B_imp = min(log(1+5), 3.0) + 0.5 (in recency) = log(6) + 0.5.

    Constructed directly via the schema (not via ProfileBuilder.apply_commit)
    so there is no incremental-decay drift baked into the numbers -- the
    golden values below are exact, not approximations.
    """
    profile = Profile()
    profile.meta.total_units = 5
    profile.explicit.append(
        ExplicitEntry(input="kansuu", surface="関数", accept_count=3, reject_count=1)
    )
    profile.unigram["猫"] = UnigramEntry(count=5.0, reading="neko", last_used=5)
    profile.recency = ["猫"]
    return profile


class ExplicitBonusTest(unittest.TestCase):
    def test_aggregates_by_surface(self):
        profile = Profile()
        profile.explicit.append(
            ExplicitEntry(input="a", surface="X", accept_count=2, reject_count=0)
        )
        profile.explicit.append(
            ExplicitEntry(input="b", surface="X", accept_count=1, reject_count=1)
        )
        bonus = compute_explicit_bonus(profile, gamma=1.0)
        b_exp, reading = bonus["X"]
        self.assertAlmostEqual(b_exp, math.log1p(3) - math.log1p(1))
        # Reading is the first entry seen for this surface.
        self.assertEqual(reading, "a")


class ImplicitBonusTest(unittest.TestCase):
    def test_caps_at_kappa_and_adds_recency_bonus(self):
        profile = Profile()
        profile.meta.total_units = 10
        profile.unigram["w"] = UnigramEntry(count=1000.0, reading="r", last_used=10)
        profile.recency = ["w"]
        builder = ProfileBuilder(profile=profile, half_life=DEFAULT_HALF_LIFE)
        bonus = compute_implicit_bonus(profile, builder, kappa=3.0, rho=0.5)
        b_imp, reading = bonus["w"]
        self.assertAlmostEqual(b_imp, 3.0 + 0.5)
        self.assertEqual(reading, "r")

    def test_no_recency_bonus_when_absent(self):
        profile = Profile()
        profile.meta.total_units = 10
        profile.unigram["w"] = UnigramEntry(count=1.0, reading=None, last_used=10)
        profile.recency = []
        builder = ProfileBuilder(profile=profile, half_life=DEFAULT_HALF_LIFE)
        bonus = compute_implicit_bonus(profile, builder, kappa=3.0, rho=0.5)
        b_imp, reading = bonus["w"]
        self.assertAlmostEqual(b_imp, math.log1p(1.0))
        self.assertEqual(reading, "")


class ExportBiasTest(unittest.TestCase):
    def test_skips_rows_with_both_bonuses_zero(self):
        profile = Profile()
        profile.explicit.append(
            ExplicitEntry(input="a", surface="zero", accept_count=0, reject_count=0)
        )
        rows = export_bias(profile)
        self.assertEqual(rows, [])

    def test_rows_sorted_by_surface(self):
        profile = Profile()
        profile.meta.total_units = 1
        profile.unigram["b"] = UnigramEntry(count=1.0, last_used=1)
        profile.unigram["a"] = UnigramEntry(count=1.0, last_used=1)
        rows = export_bias(profile)
        self.assertEqual([row[0] for row in rows], ["a", "b"])


class ExportBiasGoldenTest(unittest.TestCase):
    """Cross-repo numeric contract: this Profile must export to exactly the
    TSV in tests/fixtures/profile_bias_golden.tsv, which is byte-for-byte
    mirrored at kairo/crates/kairo-core/tests/fixtures/profile_bias_golden.tsv
    and re-parsed there by kairo-core's ProfileBias parser test.
    """

    def test_matches_golden_fixture(self):
        profile = _golden_profile()
        rows = export_bias(profile)
        actual = format_tsv(rows)

        golden_path = FIXTURES_DIR / "profile_bias_golden.tsv"
        golden_lines = []
        for line in golden_path.read_text(encoding="utf-8").splitlines():
            if line == HEADER or (line and not line.startswith("#")):
                golden_lines.append(line)
        expected = "\n".join(golden_lines) + "\n"

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
