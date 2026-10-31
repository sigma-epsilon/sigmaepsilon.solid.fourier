from collections.abc import Callable, Hashable, Iterable
from functools import partial
from types import NoneType

import numpy as np
from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math import atleast1d

from .enums import MechanicalModelType
from .postproc import postproc
from .preproc import lhs_Navier, rhs_Bernoulli
from .problem import NavierProblem
from .proc import linsolve_Bernoulli, linsolve_Timoshenko
from .protocols import LoadGroupProtocol
from .result import BeamLoadCaseResultLinStat
from .utils import check_points_in_domain, to_positive_float, to_positive_int

__all__ = ["NavierBeam"]


def _check_beam_points(
    points: float | Iterable | NoneType, length: float
) -> np.ndarray:
    """
    Returns the points of evaluation of a beam as an 1d float array, or raises
    an error if they are missing, have an invalid shape or are outside the span.
    """
    if points is None:
        raise ValueError("The points of evaluation must be provided.")
    points = atleast1d(np.asarray(points, dtype=float))
    if points.ndim != 1:
        raise ValueError(
            f"The points must be a scalar or an 1d iterable, got shape {points.shape}."
        )
    check_points_in_domain(points, length)
    return points


class NavierBeam(NavierProblem):
    """
    A class designed to handle simply-supported beams bent in the X-Y plane
    and solve them using Navier's method.

    The beam model can be either Euler-Bernoulli or Timoshenko, depending on whether shear
    stiffness is provided at instantiation.

    Parameters
    ----------
    length: float
        The length of the beam, a positive number.
    N: int, Optional
        The number of harmonic terms involved in the approximation, a positive
        integer. Default is 100.
    EI: float
        Bending stiffness, a positive number.
    GA: float, Optional
        Shear stiffness, a positive number. Only for Timoshenko beams.
        Default is None.
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
    def length(self) -> float:
        """The length of the beam."""
        return self._length

    @length.setter
    def length(self, value: float):
        """Sets the length of the beam."""
        self._length = to_positive_float(value, "length")

    @property
    def N(self) -> int:
        """The number of harmonic terms involved in the approximation."""
        return self._N

    @N.setter
    def N(self, value: int):
        """Sets the number of harmonic terms involved in the approximation."""
        self._N = to_positive_int(value, "N")

    @property
    def EI(self) -> float:
        """The bending stiffness of the beam."""
        return self._EI

    @EI.setter
    def EI(self, value: float):
        """Sets the bending stiffness of the beam."""
        self._EI = to_positive_float(value, "EI")

    @property
    def GA(self) -> float | None:
        """The shear stiffness of the beam, or None for an Euler-Bernoulli beam."""
        return self._GA

    @GA.setter
    def GA(self, value: float | None):
        """Sets the shear stiffness of the beam."""
        self._GA = None if value is None else to_positive_float(value, "GA")

    @property
    def size(self) -> float | int:
        """The length of the beam."""
        return self.length

    @size.setter
    def size(self, value: float | int):
        """Sets the length of the beam."""
        self.length = value

    @property
    def shape(self) -> int:
        """The number of harmonic terms involved in the approximation."""
        return self.N

    @shape.setter
    def shape(self, value: int):
        """Sets the number of harmonic terms involved in the approximation."""
        self.N = value

    @property
    def model_type(self) -> MechanicalModelType:
        """The mechanical model type of the beam."""
        if self.GA is None:
            return MechanicalModelType.BERNOULLI_EULER_BEAM
        else:
            return MechanicalModelType.TIMOSHENKO_BEAM

    def _check_points(self, points: float | Iterable | NoneType) -> np.ndarray:
        return _check_beam_points(points, self.length)

    def _stiffness(self) -> tuple:
        return self.EI, self.GA

    def _points_checker(self) -> Callable[[Iterable], np.ndarray]:
        return partial(_check_beam_points, length=self.length)

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
        points: float or Iterable
            A float or an 1d iterable of coordinates, where the results are
            to be evaluated. If it is a scalar, the resulting dictionary
            contains 1d arrays for every quantity, for every load case. If
            there are multiple points, the result attached to a load case is
            a 2d array, where the first axis goes along the points. The points
            must lie on the beam and they are required.

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

        points = self._check_points(points)

        # STIFFNESS
        lhs = lhs_Navier(self.length, self.N, D=self.EI, S=self.GA)

        # LOADS
        # The addresses and the cases are collected together, so that the results
        # are assigned to the cases they belong to, regardless of any other items
        # in the load group.
        case_items = list(loads.cases(return_address=True))
        if len(case_items) == 0:
            raise ValueError("The load group contains no load cases.")
        addresses, cases = zip(*case_items)
        rhs = np.stack([lc.rhs(problem=self) for lc in cases], axis=0)
        # rhs.shape = (nRHS, nMN, nComponent)
        rhs0 = np.stack([lc.rhs0(problem=self) for lc in cases], axis=0)
        # zero-order coefficients of moment loads, see LoadCase.rhs0

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
        res = postproc(
            self.length, self.N, points, coeffs, rhs, self.EI, self.GA, loads0=rhs0
        )
        # res.shape = (nLHS, nRHS, nPoint, nComponent)

        result = DeepDict()
        for i, addr in enumerate(addresses):
            result[addr] = self._postproc_linstat_load_case_result(
                res[0, i, :, :],
                points=points,
                evaluator=self._load_case_evaluator(
                    coeffs[:, i : i + 1], rhs[i : i + 1], rhs0[i : i + 1]
                ),
            )
        result.lock()

        return result
