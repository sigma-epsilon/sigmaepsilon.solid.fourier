from abc import abstractmethod
from typing import List

import numpy as np
import xarray as xr


class NavierProblem:
    """
    Base class for Navier problems. The sole reason of this class is to
    avoid circular referencing.
    """

    postproc_components: List[str]

    def _postproc_result_to_xarray_2d(self, data) -> xr.DataArray:
        nP = len(data)
        components = self.__class__.postproc_components
        coords = [np.arange(nP), components]
        xarr = xr.DataArray(
            data, coords=coords, dims=["index", "component"], name="values"
        )
        return xarr

    @abstractmethod
    def solve(self, *args, **kwargs): ...
