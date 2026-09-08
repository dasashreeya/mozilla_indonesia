"""Generation controls for Qwen's inner decoder and independent model caches."""
import functools
import math


def set_beams(model, beams):
    if beams < 1:
        raise ValueError("Beam count must be positive")
    # The outer generate() delegates to thinker.generate() without its config.
    model.thinker.generation_config.num_beams = beams
    model.thinker.generation_config.do_sample = False


def install_probability_ensemble(primary, others, weights=None):
    """Reuse Transformers generation while mixing next-token distributions.

    Each model keeps its own KV cache, reset at the start of every batch.
    Transformers' beam-reorder hook applies the same permutation to all caches.
    """
    import torch

    models = [primary.thinker] + [model.thinker for model in others]
    weights = weights or [1 / len(models)] * len(models)
    if len(weights) != len(models) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Ensemble weights must be finite, positive, and match the models")
    total = sum(weights)
    weights = [w / total for w in weights]
    if len({model.config.get_text_config().vocab_size for model in models}) != 1:
        raise ValueError("Ensemble vocabularies have different sizes")
    original = models[0].forward
    caches = [None] * len(others)

    @functools.wraps(original)
    def forward(*args, **kwargs):
        position = kwargs.get("cache_position")
        if position is None:
            raise ValueError("Ensemble requires explicit generation cache positions")
        if int(position[0]) == 0:
            caches[:] = [None] * len(others)
        output = original(*args, **kwargs)
        mixed = output.logits[:, -1:, :].float().log_softmax(-1) + math.log(weights[0])
        # Drop full-prefix logits before running another model's audio prefill.
        output.logits = mixed
        for index, model in enumerate(models[1:]):
            inputs = dict(kwargs, past_key_values=caches[index])
            extra = model(*args, **inputs)
            caches[index] = extra.past_key_values
            score = extra.logits[:, -1:, :].float().log_softmax(-1) + math.log(weights[index + 1])
            mixed = torch.logaddexp(mixed, score)
            del extra, score
        output.logits = mixed
        return output

    models[0].forward = forward

    def reorder_cache(primary_cache, beam_indices):
        primary_cache.reorder_cache(beam_indices)
        for cache in caches:
            if cache is None:
                raise RuntimeError("Secondary cache missing during beam reordering")
            cache.reorder_cache(beam_indices)
        return primary_cache

    models[0]._reorder_cache = reorder_cache
    return original
