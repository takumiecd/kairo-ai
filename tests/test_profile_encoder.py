import unittest

import torch
import torch.nn as nn

from dataset.vocab import build_output_vocab
from model.profile_encoder import DEFAULT_TOP_K
from model.profile_encoder import ProfileEncoder
from model.profile_encoder import empty_profile_snapshot
from model.profile_encoder import encode_profile_batch
from model.profile_encoder import extract_domain_vector
from model.profile_encoder import extract_lang_vector
from model.profile_encoder import select_top_k_words
from user_profile.builder import ProfileBuilder


def _profile_with_words(words: list[str]) -> dict:
    builder = ProfileBuilder()
    for word in words:
        builder.apply_commit(word)
    return builder.profile.to_dict()


class SelectTopKWordsTest(unittest.TestCase):
    def test_orders_by_decayed_count_descending(self):
        builder = ProfileBuilder(half_life=10**9)
        for _ in range(5):
            builder.apply_commit("猫")
        for _ in range(2):
            builder.apply_commit("犬")
        builder.apply_commit("鳥")
        profile = builder.profile.to_dict()

        top = select_top_k_words(profile, top_k=2)

        self.assertEqual(top, ["猫", "犬"])

    def test_respects_top_k_limit(self):
        profile = _profile_with_words(["a", "b", "c", "d"])
        self.assertEqual(len(select_top_k_words(profile, top_k=2)), 2)

    def test_empty_profile_has_no_words(self):
        self.assertEqual(select_top_k_words(empty_profile_snapshot()), [])


class DomainLangExtractionTest(unittest.TestCase):
    def test_empty_profile_domain_is_uniform(self):
        vector = extract_domain_vector(empty_profile_snapshot())
        self.assertAlmostEqual(sum(vector), 1.0, places=6)
        self.assertEqual(len(vector), 3)

    def test_empty_profile_lang_is_zero(self):
        self.assertEqual(extract_lang_vector(empty_profile_snapshot()), [0.0, 0.0])


class EncodeProfileBatchTest(unittest.TestCase):
    def setUp(self):
        self.vocab = build_output_vocab(["猫", "犬", "鳥", "abc"])

    def test_output_shapes(self):
        profiles = [_profile_with_words(["猫", "犬"]), empty_profile_snapshot()]
        batch = encode_profile_batch(profiles, self.vocab, top_k=4)

        self.assertEqual(batch["domain"].shape, (2, 3))
        self.assertEqual(batch["lang"].shape, (2, 2))
        self.assertEqual(batch["word_mask"].shape[0], 2)
        self.assertEqual(batch["word_mask"].shape[1], 4)
        self.assertEqual(batch["word_char_ids"].shape[:2], (2, 4))
        self.assertEqual(batch["word_char_mask"].shape, batch["word_char_ids"].shape)

    def test_word_mask_reflects_word_count(self):
        profiles = [_profile_with_words(["猫", "犬"])]
        batch = encode_profile_batch(profiles, self.vocab, top_k=4)
        self.assertEqual(int(batch["word_mask"][0].sum().item()), 2)

    def test_empty_profile_has_no_active_word_slots(self):
        batch = encode_profile_batch([empty_profile_snapshot()], self.vocab, top_k=4)
        self.assertEqual(int(batch["word_mask"].sum().item()), 0)


class ProfileEncoderTest(unittest.TestCase):
    def setUp(self):
        self.vocab = build_output_vocab(["猫", "犬", "鳥", "abc"])
        self.embed_dim = 6
        self.output_dim = 10
        self.embedding = nn.Embedding(len(self.vocab.id_to_token), self.embed_dim)
        self.encoder = ProfileEncoder(
            embed_dim=self.embed_dim, output_dim=self.output_dim, top_k=DEFAULT_TOP_K
        )

    def test_output_shape_is_batch_by_output_dim(self):
        profiles = [_profile_with_words(["猫", "犬"]), empty_profile_snapshot()]
        batch = encode_profile_batch(profiles, self.vocab, top_k=DEFAULT_TOP_K)

        output = self.encoder(
            domain=batch["domain"],
            lang=batch["lang"],
            word_char_ids=batch["word_char_ids"],
            word_char_mask=batch["word_char_mask"],
            word_mask=batch["word_mask"],
            embedding=self.embedding,
        )

        self.assertEqual(output.shape, (2, self.output_dim))

    def test_is_deterministic_given_same_input(self):
        profiles = [_profile_with_words(["猫", "犬", "鳥"])]
        batch = encode_profile_batch(profiles, self.vocab, top_k=DEFAULT_TOP_K)

        self.encoder.eval()
        with torch.no_grad():
            first = self.encoder(
                domain=batch["domain"],
                lang=batch["lang"],
                word_char_ids=batch["word_char_ids"],
                word_char_mask=batch["word_char_mask"],
                word_mask=batch["word_mask"],
                embedding=self.embedding,
            )
            second = self.encoder(
                domain=batch["domain"],
                lang=batch["lang"],
                word_char_ids=batch["word_char_ids"],
                word_char_mask=batch["word_char_mask"],
                word_mask=batch["word_mask"],
                embedding=self.embedding,
            )
        self.assertTrue(torch.equal(first, second))

    def test_empty_profile_differs_from_populated_profile(self):
        populated = _profile_with_words(["猫", "犬", "鳥"])
        profiles = [populated, empty_profile_snapshot()]
        batch = encode_profile_batch(profiles, self.vocab, top_k=DEFAULT_TOP_K)

        self.encoder.eval()
        with torch.no_grad():
            output = self.encoder(
                domain=batch["domain"],
                lang=batch["lang"],
                word_char_ids=batch["word_char_ids"],
                word_char_mask=batch["word_char_mask"],
                word_mask=batch["word_mask"],
                embedding=self.embedding,
            )
        self.assertFalse(torch.equal(output[0], output[1]))


if __name__ == "__main__":
    unittest.main()
