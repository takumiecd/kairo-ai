import unittest

from dataset.noise import InputNoiser


class NoiseTest(unittest.TestCase):
    def test_augment_includes_clean_input(self):
        noiser = InputNoiser(seed=0)

        variants = noiser.augment("baguwoshuuseishita", n=3)

        self.assertEqual(variants[0].text, "baguwoshuuseishita")
        self.assertEqual(variants[0].noise, "none")
        self.assertGreaterEqual(len(variants), 2)

    def test_romaji_variant_changes_known_pattern(self):
        noiser = InputNoiser(seed=3)

        noisy = noiser.apply_romaji_variant("shuuseishita")

        self.assertNotEqual(noisy, "shuuseishita")


if __name__ == "__main__":
    unittest.main()
