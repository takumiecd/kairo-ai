import unittest

import torch

from model.lora import LoRALinear
from model.lora import count_trainable_parameters
from model.lora import inject_lora
from model.lora import load_lora_state_dict
from model.lora import lora_state_dict
from model.lora import mark_only_lora_trainable
from model.transducer import KairoTransducer


def small_model() -> KairoTransducer:
    return KairoTransducer(
        input_vocab_size=10,
        output_vocab_size=12,
        input_embed_dim=8,
        output_embed_dim=8,
        encoder_hidden_dim=16,
        prediction_hidden_dim=16,
        joint_hidden_dim=16,
    )


class LoRALinearTest(unittest.TestCase):
    def test_zero_init_is_identity(self):
        base = torch.nn.Linear(6, 4)
        wrapped = LoRALinear(base, r=2, alpha=4.0)
        x = torch.randn(3, 6)
        torch.testing.assert_close(wrapped(x), base(x))

    def test_nonzero_lora_b_changes_output(self):
        base = torch.nn.Linear(6, 4)
        wrapped = LoRALinear(base, r=2, alpha=4.0)
        with torch.no_grad():
            wrapped.lora_b.add_(1.0)
        x = torch.randn(3, 6)
        self.assertFalse(torch.allclose(wrapped(x), base(x)))


class InjectLoraTest(unittest.TestCase):
    def test_targets_joint_layers_by_default(self):
        model = small_model()
        replaced = inject_lora(model, r=4, alpha=8.0)
        self.assertEqual(set(replaced), {"joint_fc1", "joint_fc2"})
        self.assertIsInstance(model.joint_fc1, LoRALinear)
        self.assertIsInstance(model.joint_fc2, LoRALinear)

    def test_forward_unchanged_right_after_injection(self):
        model = small_model().eval()
        x = torch.randint(0, 10, (2, 5))
        y = torch.randint(0, 12, (2, 4))
        before = model(x, y)
        inject_lora(model, r=4, alpha=8.0)
        after = model.eval()(x, y)
        torch.testing.assert_close(before, after)

    def test_only_lora_is_trainable(self):
        model = small_model()
        inject_lora(model, r=4, alpha=8.0)
        mark_only_lora_trainable(model)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))
        self.assertLess(count_trainable_parameters(model), sum(p.numel() for p in model.parameters()))

    def test_state_dict_roundtrip(self):
        model = small_model()
        inject_lora(model, r=4, alpha=8.0)
        with torch.no_grad():
            model.joint_fc1.lora_b.add_(0.5)
        adapter = lora_state_dict(model)
        self.assertTrue(adapter)
        self.assertTrue(all("lora_" in key for key in adapter))

        fresh = small_model()
        inject_lora(fresh, r=4, alpha=8.0)
        load_lora_state_dict(fresh, adapter)
        torch.testing.assert_close(fresh.joint_fc1.lora_b, model.joint_fc1.lora_b)


if __name__ == "__main__":
    unittest.main()
