from typing import Iterable

import numpy as np
from numpy import ndarray

from sigmaepsilon.mesh import PolyData

from .loads import LoadCase

__all__ = ["TriCollectionLoad"]


class TriCollectionLoad(LoadCase):
    """
    For loads defined over a collection of triangles.

    This class allows for arbitrary loads to be defined over a collection of
    triangles.

    Parameters
    ----------
    points_or_mesh: Iterable[float | int] | PolyData
        The points or mesh where the load is applied.
    values: Iterable[float | int] | str
        The values of the load.
    triangles: Iterable[int] | None
        The vertex indices (topology) of the triangles.

    """

    def __init__(
        self,
        points_or_mesh: Iterable[float | int] | PolyData,
        values: Iterable[float | int] | str,
        triangles: Iterable[int] | None = None,
    ):
        domain, value = None, None
        super().__init__(domain, value)
