"""Map a saved distributed sampler position when reducing the GPU count."""

import math


def map_resume_batch(old, new, next_batch, dataset_size, allow_world_size_change=False):
    old_world, new_world = old["world_size"], new["world_size"]
    old_accum, new_accum = old["accumulation"], new["accumulation"]
    if old_world == new_world:
        if old_accum != new_accum:
            raise ValueError("Resume accumulation changed without a GPU count change")
        return next_batch
    if not allow_world_size_change:
        raise ValueError("GPU count changed; use --allow-world-size-change explicitly")
    if new_world > old_world:
        raise ValueError("Resume currently supports reducing, not increasing, GPU count")
    if old_world * old_accum != new_world * new_accum:
        raise ValueError("Preserve world_size * accumulation when changing GPU count")
    old_batches = math.ceil(dataset_size / old_world)
    new_batches = math.ceil(dataset_size / new_world)
    if math.ceil(old_batches / old_accum) != math.ceil(new_batches / new_accum):
        raise ValueError("GPU change would alter steps per epoch and the LR schedule")
    if not 0 <= next_batch <= old_batches:
        raise ValueError("Saved sampler position is outside this epoch")
    # DistributedSampler assigns consecutive global permutation entries across
    # ranks. Preserve the consumed prefix, including epoch-end padding handling.
    consumed = min(next_batch * old_world, dataset_size)
    if consumed == dataset_size:
        return new_batches
    if consumed % new_world:
        raise ValueError("Saved sampler position cannot be mapped to the new GPU count")
    mapped = consumed // new_world
    if mapped % new_accum:
        raise ValueError("Saved position is not at an optimizer-step boundary")
    return mapped
