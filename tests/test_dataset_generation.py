import unittest

from dataset.generate import DatasetGenerator
from dataset.reading import split_mixed_text


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

    def test_splits_mixed_text(self):
        spans = split_mixed_text('git commit -m "修正した"')

        self.assertEqual(spans[0].kind, "literal")
        self.assertEqual(spans[1].kind, "japanese")


if __name__ == "__main__":
    unittest.main()
