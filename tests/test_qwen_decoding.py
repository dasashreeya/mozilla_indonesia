import types
import unittest

from submission_src.qwen_decoding import install_probability_ensemble, set_beams


class BeamTests(unittest.TestCase):
    def test_beams_reach_the_inner_generator(self):
        outer = types.SimpleNamespace(num_beams=1)
        inner = types.SimpleNamespace(num_beams=1, do_sample=True)
        model = types.SimpleNamespace(generation_config=outer,
                                      thinker=types.SimpleNamespace(generation_config=inner))
        set_beams(model, 4)
        self.assertEqual(inner.num_beams, 4)
        self.assertFalse(inner.do_sample)
        self.assertEqual(outer.num_beams, 1)

    def test_invalid_beam_count_rejected(self):
        with self.assertRaises(ValueError):
            set_beams(None, 0)


class EnsembleTests(unittest.TestCase):
    def test_each_model_cache_is_independent_and_resets_for_new_batch(self):
        import torch

        class Cache:
            def __init__(self):
                self.order = None

            def reorder_cache(self, order):
                self.order = order

        class Thinker:
            def __init__(self, logits):
                self.values = logits
                self.received = []
                self.generation_config = types.SimpleNamespace(num_beams=1)
                self.config = types.SimpleNamespace(
                    get_text_config=lambda: types.SimpleNamespace(vocab_size=2))

            def forward(self, cache_position, past_key_values=None):
                self.received.append(past_key_values)
                self.last_cache = Cache()
                return types.SimpleNamespace(logits=torch.tensor([[self.values]]),
                                             past_key_values=self.last_cache)

            def __call__(self, **kwargs):
                return self.forward(**kwargs)

        first = Thinker([3.0, 1.0])
        second = Thinker([0.0, 2.0])
        install_probability_ensemble(types.SimpleNamespace(thinker=first),
                                     [types.SimpleNamespace(thinker=second)], weights=[0.75, 0.25])
        result = first.forward(cache_position=torch.tensor([0]), past_key_values=object())
        expected = 0.75 * torch.tensor([3.0, 1.0]).softmax(-1) + 0.25 * torch.tensor([0.0, 2.0]).softmax(-1)
        torch.testing.assert_close(result.logits[0, 0].exp(), expected)
        self.assertIsNone(second.received[0])
        first.forward(cache_position=torch.tensor([1]), past_key_values=object())
        self.assertIsNotNone(second.received[1])
        first.forward(cache_position=torch.tensor([0]), past_key_values=object())
        self.assertIsNone(second.received[2])
        order = torch.tensor([2, 0, 1])
        first._reorder_cache(first.last_cache, order)
        self.assertIs(first.last_cache.order, order)
        self.assertIs(second.last_cache.order, order)
