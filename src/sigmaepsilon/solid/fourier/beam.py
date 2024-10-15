from typing import Iterable, Hashable
from types import NoneType

import numpy as np

from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math import atleast1d

from .problem import NavierProblem
from .preproc import lhs_Navier, rhs_Bernoulli
from .postproc import postproc
from .proc import linsolve_Bernoulli, linsolve_Timoshenko
from .result import BeamLoadCaseResultLinStat
from .protocols import LoadGroupProtocol

__all__ = ["NavierBeam"]


class NavierBeam(NavierProblem):
    """
    A class designed to handle simply-supported plates bent in the X-Y plane
    and solve them using Navier's method. The beam model can be either
    Euler-Bernoulli or Timoshenko, depending on whether shear stiffness is
    provided at instantiation.

    Parameters
    ----------
    length: float
        The length of the beam.
    N: int, Optional
        The number of harmonic terms involved in the approximation.
        Default is 100.
    EI: float
        Bending stiffness.
    GA: float, Optional
        Shear stiffness. Only for Timoshenko beams. Default is None.
    loads: LoadGroup, Optional
        The loads. Default is None.

    Examples
    --------
    To define an Euler-Bernoulli beam of length 10.0 and
    bending stiffness 2000.0 with 100 harmonic terms involved:

    >>> from sigmaepsilon.solid.fourier import NavierBeam
    >>> beam = NavierBeam(10.0, 100, EI=2000.0)

    To define a Timoshenko beam of length 10.0, bending stiffness
    2000.0 and shear stiffness 1500.0 with 100 harmonic terms involved:

    >>> from sigmaepsilon.solid.fourier import NavierBeam
    >>> beam = NavierBeam(10.0, 100, EI=2000.0, GA=1500.0)

    """

    result_class = BeamLoadCaseResultLinStat

    def __init__(
        self,
        length: float,
        N: int = 100,
        *,
        EI: float,
        GA: float | None = None,
        loads: LoadGroupProtocol | None = None,
    ):
        super().__init__(loads=loads)
        self.length = length
        self.EI = EI
        self.GA = GA
        self.N = N

    @property
    def size(self) -> float | int:
        return self.length

    @property
    def shape(self) -> int:
        return self.N

    def linear_static_analysis(
        self,
        *args,
        points: float | Iterable | NoneType = None,
        loads: LoadGroupProtocol | NoneType = None,
    ) -> DeepDict[Hashable, DeepDict | BeamLoadCaseResultLinStat]:
        """
        Performs a linear static analysis and calculates all postprocessing quantities at
        one ore more points.

        Parameters
        ----------
        *args: LoadGroup, float or Iterable
            The loads and the points, in any order.
        loads: :class:`~sigmaepsilon.solid.fourier.loads.LoadGroup`, Optional
            The loads.
        points: float or Iterable, Optional
            A float or an 1d iterable of coordinates, where the results are
            to be evaluated. If it is a scalar, the resulting dictionary
            contains 1d arrays for every quantity, for every load case. If
            there are multiple points, the result attached to a load case is
            a 2d array, where the first axis goes along the points.

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
                elif isinstance(arg, (float, Iterable)):
                    if points is not None:
                        raise ValueError("The points are already provided.")

                    points = arg
                else:
                    raise TypeError(f"Invalid argument type {type(arg)}.")

        loads = self.loads if loads is None else loads

        if not isinstance(loads, LoadGroupProtocol):  # pragma: no cover
            raise TypeError("The loads must be an instance of LoadGroup.")

        # STIFFNESS
        lhs = lhs_Navier(self.length, self.N, D=self.EI, S=self.GA)

        # LOADS
        load_cases = list(loads.cases())
        rhs = np.vstack(list(lc.rhs(problem=self) for lc in load_cases))

        # SOLUTION
        if self.GA is None:
            _RHS = rhs_Bernoulli(rhs, self.length)
            coeffs = linsolve_Bernoulli(lhs, _RHS)
            del _RHS
            # coeffs.shape = (nLHS, nRHS, nMN)
        else:
            coeffs = linsolve_Timoshenko(lhs, rhs)
            # coeffs.shape = (nLHS, nRHS, nMN, 2)

        # POSTPROCESSING
        points = atleast1d(points)
        res = postproc(self.length, self.N, points, coeffs, rhs, self.EI, self.GA)
        # res.shape = (nLHS, nRHS, nPoint, nComponent)

        result = DeepDict()
        for i, (addr, _) in enumerate(loads.items(deep=True, return_address=True)):
            result[addr] = self._postproc_linstat_load_case_result(res[0, i, :, :])
        result.lock()

        return result
