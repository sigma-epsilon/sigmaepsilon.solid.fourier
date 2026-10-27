"""
Monte Carlo sampling of the integration domains of loads.
"""

from collections.abc import Callable, Iterator

import numpy as np
from numpy import ndarray

from .utils import RNGType, to_positive_int

__all__ = ["monte_carlo_samples"]


def monte_carlo_samples(
    sampler: Callable[..., ndarray],
    num_samples: int,
    batch_size: int,
    measure: float,
    rng: RNGType = None,
) -> Iterator[tuple[ndarray, ndarray]]:
    """
    Yields batches of uniformly distributed random points and their weights,
    such that the weighted sum of the values of a function over all batches
    estimates its integral over the domain.

    Parameters
    ----------
    sampler: Callable
        A function that is called as ``sampler(n, rng=rng)`` and returns `n`
        uniformly distributed random points of the domain.
    num_samples: int
        The total number of samples.
    batch_size: int
        The largest number of samples in a batch. The last batch holds the
        remaining samples, however few they are.
    measure: float
        The length or the area of the domain.
    rng: int or numpy.random.Generator, Optional
        The seed or the random generator, see :func:`numpy.random.default_rng`.
        The global random state is left untouched.
    """
    num_samples = to_positive_int(num_samples, "num_mc")
    batch_size = to_positive_int(batch_size, "batch_size")
    rng = np.random.default_rng(rng)
    weight = measure / num_samples
    start = 0
    while start < num_samples:
        n = min(batch_size, num_samples - start)
        yield sampler(n, rng=rng), np.full(n, weight)
        start += n
