from typing import Tuple, Iterable

import numpy as np

from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math import atleast2d

from .problem import NavierProblem
from .loads import LoadGroup
from .preproc import lhs_Navier, rhs_Kirchhoff
from .postproc import postproc
from .proc import linsolve_Kirchhoff, linsolve_Mindlin
from .result import PlateLoadCaseResultLinStat

__all__ = ["NavierPlate"]


class NavierPlate(NavierProblem):
    """
    A class to handle semi-analytic solutions of rectangular plates with
    specific boudary conditions.

    Parameters
    ----------
    size: tuple[float]
        The size of the rectangle.
    shape: tuple[int]
        Numbers of harmonic terms involved in both directions.
    """

    result_class = PlateLoadCaseResultLinStat

    def __init__(
        self,
        size: Tuple[float],
        shape: Tuple[int],
        *,
        D: Iterable,
        S: Iterable | None = None,
    ):
        super().__init__()
        self.size = np.array(size, dtype=float)
        self.shape = np.array(shape, dtype=int)
        self.D = np.array(D, dtype=float)
        self.S = None if S is None else np.array(S, dtype=float)

    def linear_static_analysis(self, loads: LoadGroup, points: Iterable) -> DeepDict:
        """
        Performs a linear static analysis and calculates all entities at the specified points.

        Parameters
        ----------
        loads: :class:`~sigmaepsilon.solid.fourier.loads.LoadGroup`
            The loads.
        points: Iterable[Iterable[float]]
            2d float array of coordinates, where the results are to be evaluated.

        Returns
        -------
        :class:`~sigmaepsilon.deepdict.deepdict.DeepDict`
            A dictionary with the same layout as the loads.
        """
        if not isinstance(loads, LoadGroup):
            raise TypeError("The loads must be an instance of LoadGroup.")

        # STIFFNESS
        lhs = lhs_Navier(self.size, self.shape, D=self.D, S=self.S)

        # LOADS
        load_cases = list(loads.cases())
        rhs = np.vstack(list(lc.rhs(problem=self) for lc in load_cases))

        # SOLUTION
        if self.S is None:
            _rhs = rhs_Kirchhoff(rhs, self.size)
            coeffs = linsolve_Kirchhoff(lhs, _rhs)
            del _rhs
            # coeffs.shape = (nLHS, nRHS, nMN)
        else:
            coeffs = linsolve_Mindlin(lhs, rhs)
            # coeffs.shape = (nLHS, nRHS, nMN, 3)

        # POSTPROCESSING
        points = atleast2d(points)
        res = postproc(self.size, self.shape, points, coeffs, rhs, self.D, self.S)
        # res.shape = (nLHS, nRHS, nP, nX)

        result = DeepDict()
        for i, (addr, _) in enumerate(loads.items(deep=True, return_address=True)):
            result[addr] = self._postproc_linstat_load_case_result(res[0, i, :, :])
        result.lock()

        return result
