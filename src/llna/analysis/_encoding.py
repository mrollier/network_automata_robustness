"""Numpy density -> interval one-hot encoding (float64 twin of the torch
encoding in llna.automata.LLNA.interval_encoding; boundary behaviour of the
two can differ at float32-representable interval edges, see the
characterization fixtures)."""

import numpy as np


def _interval_encoding(resolution: int, rhos, iso=True):
    if not iso:  # classic psuedo-isomorphic case
        belongs_to = lambda x, k: (k <= resolution * x) & (resolution * x < k + 1)
        is_boundary = lambda x, k: (k == resolution - 1) & (x == 1)
        return np.stack([(belongs_to(rhos, k) | is_boundary(rhos, k)) for k in range(resolution)], 2)
    else:  # altered isomorphic case
        if resolution % 2 == 0:
            raise ValueError("Resolution must be an odd number if iso=True.")
        belongs_to_lower = lambda x, k: (
            (k < resolution / 2) & (x >= k / resolution) & (x < (k + 1) / resolution)
        )
        belongs_to_middle = lambda x, k: (
            (k == (resolution - 1) / 2) & (x >= k / resolution) & (x <= (k + 1) / resolution)
        )
        belongs_to_upper = lambda x, k: (
            (k > resolution / 2) & (x > k / resolution) & (x <= (k + 1) / resolution)
        )
        return np.stack(
            [
                (belongs_to_lower(rhos, k) | belongs_to_middle(rhos, k) | belongs_to_upper(rhos, k))
                for k in range(resolution)
            ],
            2,
        )
