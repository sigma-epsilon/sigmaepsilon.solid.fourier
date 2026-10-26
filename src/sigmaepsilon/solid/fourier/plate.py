from typing import Iterable, Hashable
from types import NoneType

import numpy as np
from numpy import ndarray

from sigmaepsilon.deepdict import DeepDict

from .problem import NavierProblem
from .preproc import lhs_Navier, rhs_Kirchhoff
from .postproc import postproc
from .proc import linsolve_Kirchhoff, linsolve_Mindlin
from .result import PlateLoadCaseResultLinStat
from .protocols import LoadGroupProtocol
from .enums import MechanicalModelType
from .utils import to_positive_float, to_positive_int, check_points_in_domain

__all__ = ["NavierPlate"]


# relative tolerance of the checks on the structure of the stiffness matrices
STIFFNESS_TOLERANCE = 1e-8


def _check_stiffness_matrix(value: Iterable, name: str, size: int) -> ndarray:
    matrix = np.array(value, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(
            f"'{name}' must be a {size}x{size} matrix, got shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"'{name}' must be finite.")
    tol = STIFFNESS_TOLERANCE * np.max(np.abs(matrix))
    if np.any(np.abs(matrix - matrix.T) > tol):
        raise ValueError(f"'{name}' must be symmetric.")
    if np.any(np.diag(matrix) <= 0):
        raise ValueError(f"The diagonal of '{name}' must be positive.")
    return matrix


def _check_bending_stiffness(value: Iterable) -> ndarray:
    D = _check_stiffness_matrix(value, "D", 3)
    tol = STIFFNESS_TOLERANCE * np.max(np.abs(D))
    if abs(D[0, 2]) > tol or abs(D[1, 2]) > tol:
        raise ValueError(
            "The Navier solution is only valid for specially orthotropic plates, "
            "the bending-twisting coupling terms D16 and D26 must be zero, got "
            f"D16={D[0, 2]} and D26={D[1, 2]}."
        )
    return D


def _check_shear_stiffness(value: Iterable) -> ndarray:
    S = _check_stiffness_matrix(value, "S", 2)
    if abs(S[0, 1]) > STIFFNESS_TOLERANCE * np.max(np.abs(S)):
        raise ValueError(
            "The Navier solution is only valid for specially orthotropic plates, "
            f"the off-diagonal term of the shear stiffness must be zero, got {S[0, 1]}."
        )
    return S


class NavierPlate(NavierProblem):
    """
    A class to handle semi-analytic solutions of simply-supported Kirchhoff-Love
    and Mindlin-Reissner rectangular plates of constant thickness.

    Parameters
    ----------
    size: tuple[float]
        The size of the rectangle, two positive numbers.
    shape: tuple[int]
        Numbers of harmonic terms involved in both directions, two positive integers.
    D: Iterable
        3x3 flexural constitutive matrix. It must be symmetric with a positive
        diagonal, and the coupling terms D16 and D26 must be zero.
    S: Iterable, Optional
        2x2 shear constitutive matrix. Only for Mindlin-Reissner plates. It must be
        diagonal with positive entries. Default is None.
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
        if not isinstance(size, Iterable) or len(size) != 2:
            raise ValueError(f"'size' must be a pair of numbers, got {size!r}.")
        if not isinstance(shape, Iterable) or len(shape) != 2:
            raise ValueError(f"'shape' must be a pair of integers, got {shape!r}.")
        self._size = np.array([to_positive_float(v, "size") for v in size])
        self._shape = np.array([to_positive_int(v, "shape") for v in shape])
        self.D = D
        self.S = S

    @property
    def D(self) -> ndarray:
        """Returns the bending stiffness matrix."""
        return self._D

    @D.setter
    def D(self, value: Iterable):
        """Sets the bending stiffness matrix."""
        self._D = _check_bending_stiffness(value)

    @property
    def S(self) -> ndarray | None:
        """Returns the shear stiffness matrix, or None for a Kirchhoff-Love plate."""
        return self._S

    @S.setter
    def S(self, value: Iterable | None):
        """Sets the shear stiffness matrix."""
        self._S = None if value is None else _check_shear_stiffness(value)

    @property
    def size(self) -> ndarray[float]:
        """Returns the size of the rectangle."""
        return self._size

    @property
    def shape(self) -> ndarray[int]:
        """Returns the shape (number of harmonic terms) in both directions."""
        return self._shape

    @property
    def model_type(self) -> MechanicalModelType:
        """Returns the mechanical model type of the plate."""
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
        points: Iterable[Iterable[float]]
            2d float array of coordinates, where the results are to be evaluated.
            The points must lie on the plate and they are required.

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

        if points is None:
            raise ValueError("The points of evaluation must be provided.")
        points = check_points_in_domain(points, self.size)

        # STIFFNESS
        lhs = lhs_Navier(self.size, self.shape, D=self.D, S=self.S)

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
        if self.S is None:
            _rhs = rhs_Kirchhoff(rhs, self.size, self.shape)
            coeffs = linsolve_Kirchhoff(lhs, _rhs)
            del _rhs
            # coeffs.shape = (nLHS, nRHS, nMN)
        else:
            coeffs = linsolve_Mindlin(lhs, rhs)
            # coeffs.shape = (nLHS, nRHS, nMN, 3)

        # POSTPROCESSING
        res = postproc(
            self.size, self.shape, points, coeffs, rhs, self.D, self.S, loads0=rhs0
        )
        # res.shape = (nLHS, nRHS, nP, nX)

        result = DeepDict()
        for i, addr in enumerate(addresses):
            result[addr] = self._postproc_linstat_load_case_result(res[0, i, :, :])
        result.lock()

        return result
