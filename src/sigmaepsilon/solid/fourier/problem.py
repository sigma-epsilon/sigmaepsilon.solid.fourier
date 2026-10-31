from abc import abstractmethod
from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
from numpy import ndarray

from .postproc import postproc
from .protocols import LoadGroupProtocol
from .result import LoadCaseResultLinStat

__all__ = ["NavierProblem"]


def _copy(value):
    """
    Returns a copy of an array and any other value as it is. Scalars must stay
    scalars, since the postprocessing tells beams from plates by them.
    """
    return np.array(value) if isinstance(value, ndarray) else value


def _evaluate_load_case(
    postproc_case: Callable[[ndarray], ndarray],
    check_points: Callable[[Iterable], ndarray],
    points: Iterable,
) -> tuple[ndarray, ndarray]:
    """
    Evaluates the solution of a single load case at arbitrary points and returns
    the checked points and the results at the points, the latter as a 2d array of
    shape (nP, nComponent). Since the solution is a Fourier series, the results
    are exact at any point of the domain, there is no interpolation involved.
    """
    points = check_points(points)
    return points, postproc_case(points)[0, 0, :, :]


class NavierProblem:
    """
    Base class for Navier problems. The sole reason of this class is to
    avoid circular referencing.
    """

    result_class = LoadCaseResultLinStat
    size: float | ndarray
    shape: int | ndarray

    def __init__(self, *, loads: LoadGroupProtocol | None = None):
        self._loads = loads

    @property
    def loads(self) -> LoadGroupProtocol | None:
        """
        Returns the loads.
        """
        return self._loads

    @loads.setter
    def loads(self, value: LoadGroupProtocol | None):
        """
        Sets the loads.
        """
        self._loads = value

    def _postproc_linstat_load_case_result(
        self,
        data,
        *,
        points: ndarray | None = None,
        evaluator: Callable | None = None,
    ) -> LoadCaseResultLinStat:
        res = self.result_class(data, name="values", points=points, evaluator=evaluator)
        return res

    def _stiffness(self) -> tuple:
        """
        Returns the bending and the shear stiffness of the problem, the latter
        being `None` if shear deformations are neglected.
        """
        raise NotImplementedError

    def _points_checker(self) -> Callable[[Iterable], ndarray]:
        """
        Returns a function that checks the points of evaluation and returns them
        in the form the postprocessing expects. The function must not depend on
        the instance, so that later changes of the problem do not affect it.
        """
        raise NotImplementedError

    def _load_case_evaluator(
        self, coeffs: ndarray, rhs: ndarray, rhs0: ndarray
    ) -> Callable[[Iterable], tuple[ndarray, ndarray]]:
        """
        Returns a function that evaluates the solution of a single load case at
        arbitrary points, see :func:`_evaluate_load_case`. The arguments are the
        coefficients of the load case, with a leading axis of length one.

        The function keeps its own copy of everything it needs, so later changes
        of the problem do not affect it.
        """
        bending, shear = self._stiffness()
        postproc_case = partial(
            postproc,
            _copy(self.size),
            _copy(self.shape),
            solution=np.array(coeffs),
            loads=np.array(rhs),
            D=_copy(bending),
            S=_copy(shear),
            loads0=np.array(rhs0),
        )
        return partial(_evaluate_load_case, postproc_case, self._points_checker())

    @abstractmethod
    def linear_static_analysis(self, *args, **kwargs):
        pass
