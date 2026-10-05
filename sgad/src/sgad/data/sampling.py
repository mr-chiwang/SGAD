# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR distributed dataset splitting for SGAD region matching.

import numpy as np

def get_local_split(items: list, world_size: int, rank: int, seed: int):
    """Shuffle and pad items deterministically, then return one rank's slice."""
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items)
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(
            items,
            world_size - (n_items % world_size),
            replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size
    local_items = padded_items[n_per_rank * rank: n_per_rank * (rank+1)]

    return local_items
