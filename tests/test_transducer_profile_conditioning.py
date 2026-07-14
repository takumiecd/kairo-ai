import unittest

import torch

from dataset.vocab import build_output_vocab
from model.profile_encoder import encode_profile_batch
from model.transducer import KairoTransducer
from user_profile.builder import ProfileBuilder

# nn.TransformerEncoder の nested-tensor fast path は eval モードでも
# 呼び出しごとに数値が揺れることがある(PyTorch 側の既知の挙動で、この
# 変更とは無関係)。bit-exact な後方互換テストのため無効化しておく。
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)


def _make_profile_features(batch_size: int, vocab, top_k: int = 8) -> dict:
    profiles = []
    for _ in range(batch_size):
        builder = ProfileBuilder()
        builder.apply_commit("猫")
        builder.apply_commit("犬")
        profiles.append(builder.profile.to_dict())
    return encode_profile_batch(profiles, vocab, top_k=top_k)


class BackwardCompatibilityTest(unittest.TestCase):
    """profile_conditioning=False は既存の挙動・重み名を完全に維持する。"""

    def _run(self, encoder_type: str, prediction_type: str) -> None:
        torch.manual_seed(0)
        model = KairoTransducer(
            input_vocab_size=12,
            output_vocab_size=15,
            embed_dim=8,
            hidden_dim=16,
            encoder_type=encoder_type,
            prediction_type=prediction_type,
        )
        self.assertFalse(hasattr(model, "profile_encoder"))
        self.assertFalse(hasattr(model, "profile_lstm_proj"))

        x = torch.randint(0, 12, (2, 5))
        y = torch.randint(0, 15, (2, 4))

        model.eval()
        with torch.no_grad():
            logits_no_kw = model(x, y)
            logits_explicit_none = model(x, y, profile_features=None)

        self.assertTrue(torch.equal(logits_no_kw, logits_explicit_none))

    def test_lstm_lstm(self):
        self._run("lstm", "lstm")

    def test_transformer_transformer(self):
        self._run("transformer", "transformer")

    def test_bit_exact_against_pre_profile_reference(self):
        """profile_conditioning 引数を渡さない旧来の呼び出しと完全一致する。"""
        torch.manual_seed(42)
        model_a = KairoTransducer(
            input_vocab_size=10, output_vocab_size=12, embed_dim=8, hidden_dim=16
        )
        torch.manual_seed(42)
        model_b = KairoTransducer(
            input_vocab_size=10,
            output_vocab_size=12,
            embed_dim=8,
            hidden_dim=16,
            profile_conditioning=False,
        )
        x = torch.randint(0, 10, (2, 4))
        y = torch.randint(0, 12, (2, 3))
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            out_a = model_a(x, y)
            out_b = model_b(x, y)
        self.assertTrue(torch.equal(out_a, out_b))


class ProfileConditioningForwardTest(unittest.TestCase):
    def _run(self, encoder_type: str, prediction_type: str) -> None:
        vocab = build_output_vocab(["猫", "犬", "鳥", "a"])
        torch.manual_seed(0)
        model = KairoTransducer(
            input_vocab_size=12,
            output_vocab_size=len(vocab.id_to_token),
            embed_dim=8,
            hidden_dim=16,
            encoder_type=encoder_type,
            prediction_type=prediction_type,
            profile_conditioning=True,
            profile_top_k=8,
        )
        self.assertTrue(hasattr(model, "profile_encoder"))

        batch_size = 2
        x = torch.randint(0, 12, (batch_size, 5))
        y = torch.randint(0, len(vocab.id_to_token), (batch_size, 4))
        profile_features = _make_profile_features(batch_size, vocab, top_k=8)

        logits = model(x, y, profile_features=profile_features)
        self.assertEqual(logits.shape[:3], (batch_size, 5, 4))

        loss = logits.sum()
        loss.backward()
        # プロファイルエンコーダの勾配が実際に流れている(死んだ経路でない)ことを確認。
        grad = model.profile_encoder.mlp[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum().item()), 0.0)

    def test_lstm_prediction_network(self):
        self._run("lstm", "lstm")

    def test_transformer_prediction_network(self):
        self._run("transformer", "transformer")

    def test_missing_profile_features_falls_back_to_unconditioned(self):
        vocab = build_output_vocab(["猫", "犬"])
        torch.manual_seed(0)
        model = KairoTransducer(
            input_vocab_size=10,
            output_vocab_size=len(vocab.id_to_token),
            embed_dim=8,
            hidden_dim=16,
            profile_conditioning=True,
        )
        x = torch.randint(0, 10, (2, 4))
        y = torch.randint(0, len(vocab.id_to_token), (2, 3))
        # profile_features=None でも例外を投げず、プレフィックス無しの経路を通る。
        logits = model(x, y)
        self.assertEqual(logits.shape[:3], (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
