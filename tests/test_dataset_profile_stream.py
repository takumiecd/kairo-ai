import unittest

from dataset.generate import DatasetGenerator
from dataset.profile_stream import PersonaSource
from dataset.profile_stream import generate_persona_examples
from dataset.profile_stream import parse_persona_arg


class ParsePersonaArgTest(unittest.TestCase):
    def test_parses_name_and_path(self):
        source = parse_persona_arg("engineer=docs/corpus.txt")
        self.assertEqual(source, PersonaSource(name="engineer", path="docs/corpus.txt"))

    def test_parses_domain_and_format(self):
        source = parse_persona_arg("novelist=aozora.zip:prose:aozora")
        self.assertEqual(
            source,
            PersonaSource(name="novelist", path="aozora.zip", domain="prose", format="aozora"),
        )

    def test_missing_equals_raises(self):
        with self.assertRaises(ValueError):
            parse_persona_arg("no-equals-sign")


class GeneratePersonaExamplesTest(unittest.TestCase):
    def setUp(self):
        self.generator = DatasetGenerator(seed=0)
        self.source = PersonaSource(name="engineer", path="unused", domain="code")

    def _sentences(self):
        return [
            {"text": "猫が歩く。", "domain": "prose"},
            {"text": "犬が走る。", "domain": "prose"},
            {"text": "鳥が飛ぶ。", "domain": "prose"},
        ]

    def test_record_shape(self):
        examples = generate_persona_examples(
            self.source,
            self._sentences(),
            self.generator,
            snapshot_every=1,
            synthetic_explicit_rate=0.0,
            lookahead=5,
            seed=0,
            context_window=1,
        )
        self.assertEqual(len(examples), 3)
        first = examples[0]
        self.assertEqual(first.persona, "engineer")
        self.assertEqual(first.target, "猫が歩く。")
        self.assertIsInstance(first.input, str)
        self.assertGreater(len(first.input), 0)
        self.assertEqual(first.context, "")  # 最初の文には前の文脈がない
        self.assertIn("meta", first.profile)
        self.assertEqual(first.profile["meta"]["total_units"], 0)  # u_0 は空

    def test_context_window_carries_previous_sentence(self):
        examples = generate_persona_examples(
            self.source,
            self._sentences(),
            self.generator,
            snapshot_every=1,
            synthetic_explicit_rate=0.0,
            lookahead=5,
            seed=0,
            context_window=1,
        )
        self.assertEqual(examples[1].context, "猫が歩く。")
        self.assertEqual(examples[2].context, "犬が走る。")

    def test_context_window_zero_disables_context(self):
        examples = generate_persona_examples(
            self.source,
            self._sentences(),
            self.generator,
            snapshot_every=1,
            synthetic_explicit_rate=0.0,
            lookahead=5,
            seed=0,
            context_window=0,
        )
        self.assertTrue(all(example.context == "" for example in examples))

    def test_profile_snapshots_grow_across_stream(self):
        examples = generate_persona_examples(
            self.source,
            self._sentences(),
            self.generator,
            snapshot_every=1,
            synthetic_explicit_rate=0.0,
            lookahead=5,
            seed=0,
            context_window=1,
        )
        totals = [example.profile["meta"]["total_units"] for example in examples]
        self.assertEqual(totals, sorted(totals))
        self.assertGreater(totals[-1], totals[0])

    def test_independent_personas_start_from_fresh_profile(self):
        """複数ペルソナは独立した仮想ユーザー(u_0から開始)である。"""
        examples_a = generate_persona_examples(
            PersonaSource(name="a", path="unused"),
            self._sentences(),
            DatasetGenerator(seed=1),
            snapshot_every=1,
            synthetic_explicit_rate=0.0,
            lookahead=5,
            seed=1,
            context_window=1,
        )
        examples_b = generate_persona_examples(
            PersonaSource(name="b", path="unused"),
            self._sentences(),
            DatasetGenerator(seed=2),
            snapshot_every=1,
            synthetic_explicit_rate=0.0,
            lookahead=5,
            seed=2,
            context_window=1,
        )
        self.assertEqual(examples_a[0].profile["meta"]["total_units"], 0)
        self.assertEqual(examples_b[0].profile["meta"]["total_units"], 0)


if __name__ == "__main__":
    unittest.main()
