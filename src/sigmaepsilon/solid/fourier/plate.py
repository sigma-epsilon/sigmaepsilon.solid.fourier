from typing import Iterable, Hashable
from types import NoneType

import numpy as np

from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math import atleast2d

from .problem import NavierProblem
from .preproc import lhs_Navier, rhs_Kirchhoff
from .postproc import postproc
from .proc import linsolve_Kirchhoff, linsolve_Mindlin
from .result import PlateLoadCaseResultLinStat
from .protocols import LoadGroupProtocol
from .enums import MechanicalModelType

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
    D: Iterable
        3x3 flexural constitutive matrix.
    S: Iterable, Optional
        2x2 shear constitutive matrix.
    loads: :class:`~sigmaepsilon.solid.fourier.loads.LoadGroup`, Optional
        The loads. Default is None.
    """

    result_class = PlateLoadCaseResultLinStat

    def __init__(
        self,
        size: tuple[float],
        shape: tuple[int],
        *,
        D: Iterable,
        S: Iterable | None = None,
        loads: LoadGroupProtocol | None = None,
    ):
        super().__init__(loads=loads)
        self._size = np.array(size, dtype=float)
        self._shape = np.array(shape, dtype=int)
        self.D = np.array(D, dtype=float)
        self.S = None if S is None else np.array(S, dtype=float)

    @property
    def size(self) -> float | int:
        return self._size

    @property
    def shape(self) -> int:
        return self._shape
    
    @property
    def model_type(self) -> MechanicalModelType:
        if self.S is None:
            return MechanicalModelType.KIRCHHOFF_LOVE_PLATE
        else:
            return MechanicalModelType.UFLYAND_MINDLIN_PLATE

    def linear_static_analysis(
        self,
        *args,
        points: Iterable | NoneType = None,
        loads: LoadGroupProtocol | NoneType = None,
    ) -> DeepDict[Hashable, DeepDict | PlateLoadCaseResultLinStat]:
        """
        Performs a linear static analysis and calculates all postprocessing quantities at
        one ore more points.

        Parameters
        ----------
        *args: Iterable[float] or :class:`~sigmaepsilon.solid.fourier.loads.LoadGroup`
            The loads and points, in any order.
        loads: :class:`~sigmaepsilon.solid.fourier.loads.LoadGroup`, Optional
            The loads.
        points: Iterable[Iterable[float]], Optional
            2d float array of coordinates, where the results are to be evaluated.

        Returns
        -------
        :class:`~sigmaepsilon.deepdict.deepdict.DeepDict`
            A dictionary with the same layout as the loads.

        """
        if len(args) > 0:

            if len(args) > 2:
                raise ValueError("Too many positional arguments.")

            for arg in args:
                if isinstance(arg, LoadGroupProtocol):
                    if loads is not None:
                        raise ValueError("The loads are already provided.")

                    loads = arg
                elif isinstance(arg, Iterable):
                    if points is not None:
                        raise ValueError("The points are already provided.")

                    points = arg
                else:
                    raise TypeError(f"Invalid argument type {type(arg)}.")

        loads = self.loads if loads is None else loads

        if not isinstance(loads, LoadGroupProtocol):  # pragma: no cover
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
