import unittest

from dataset.generate import DatasetGenerator
from dataset.reading import JapaneseRomajiConverter, split_mixed_text


class DatasetGenerationTest(unittest.TestCase):
    def test_preserves_ascii_command_fragments(self):
        generator = DatasetGenerator(seed=0)

        input_text, target = generator.generate_pair('git commit -m "バグを修正した"')

        self.assertEqual(target, 'git commit -m "バグを修正した"')
        self.assertTrue(input_text.startswith('git commit -m "'))
        self.assertIn("baguwoshuuseishita", input_text)
        self.assertNotIn("jiiaiteii", input_text)

    def test_mixed_identifier_and_japanese_text(self):
        generator = DatasetGenerator(seed=0)

        input_text, target = generator.generate_pair("user_id の型を修正した")

        self.assertEqual(target, "user_id の型を修正した")
        self.assertTrue(input_text.startswith("user_id no"))
        self.assertIn("katawoshuuseishita", input_text)

    def test_generates_clean_and_noisy_examples(self):
        generator = DatasetGenerator(seed=1)

        examples = generator.generate_examples(["ログ出力を追加した"], num_augmentations=2)

        self.assertGreaterEqual(len(examples), 2)
        self.assertEqual(examples[0].noise, "none")
        self.assertEqual(examples[0].input, examples[0].clean_input)
        self.assertTrue(all(example.target == "ログ出力を追加した" for example in examples))

    def test_noise_preserves_literal_code_spans(self):
        generator = DatasetGenerator(seed=0)

        examples = generator.generate_examples(
            ['git commit -m "バグを修正した"'],
            num_augmentations=5,
        )

        for example in examples:
            self.assertTrue(example.input.startswith('git commit -m "'))
            self.assertTrue(example.input.endswith('"'))

    def test_noise_can_touch_mutable_english_spans(self):
        generator = DatasetGenerator(seed=0)

        examples = generator.generate_examples(
            ["tokenizer は後で追加する"],
            num_augmentations=8,
        )

        self.assertTrue(any(not example.input.startswith("tokenizer ") for example in examples))

    def test_can_disable_literal_noise(self):
        generator = DatasetGenerator(seed=0, noise_literals=False)

        examples = generator.generate_examples(
            ["tokenizer は後で追加する"],
            num_augmentations=5,
        )

        self.assertTrue(all(example.input.startswith("tokenizer ") for example in examples))

    def test_long_vowel_mark_becomes_dash_key(self):
        # IME users type the long vowel ー with the '-' key (ループ -> ru-pu),
        # not as a doubled vowel (ruupu). The dataset must match real input.
        converter = JapaneseRomajiConverter()

        self.assertEqual(converter.convert_text("ループ"), "ru-pu")
        self.assertEqual(converter.convert_text("サーバーエラー"), "sa-ba-era-")
        self.assertNotIn("uu", converter.convert_text("ループ"))

    def test_wapuro_romaji_for_du_di_and_middle_dot(self):
        # pykakasi's Hepburn (zu/ji) collides with ず/じ; IME types du/di for
        # づ/ぢ. The middle dot ・ is typed with the '/' key, not left full-width.
        converter = JapaneseRomajiConverter()

        self.assertEqual(converter.convert_text("続く"), "tsuduku")
        self.assertEqual(converter.convert_text("データ・ベース"), "de-ta/be-su")
        self.assertNotIn("・", converter.convert_text("データ・ベース"))

    def test_splits_mixed_text(self):
        spans = split_mixed_text('git commit -m "修正した"')

        self.assertEqual(spans[0].kind, "literal")
        self.assertEqual(spans[1].kind, "japanese")


if __name__ == "__main__":
    unittest.main()
