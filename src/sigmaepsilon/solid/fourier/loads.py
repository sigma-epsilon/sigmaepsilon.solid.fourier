from typing import Iterable, Any, Optional, TypeVar, Generic, TypeAlias, Hashable
from numbers import Number
from abc import abstractmethod

import numpy as np
from numpy import ndarray, average as avg

from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math.function import Function

from .preproc import rhs_rect_const, rhs_conc_1d, rhs_conc_2d, rhs_line_const
from .utils import points_to_rectangle_region, sin1d, cos1d
from .protocols import NavierProblemProtocol, LoadGroupProtocol, LoadCaseProtocol

__all__ = ["LoadGroup", "LoadCase", "RectangleLoad", "LineLoad", "PointLoad"]


Float1d: TypeAlias = Iterable[float]
Float2d: TypeAlias = Iterable[Float1d]

LoadDomainType = TypeVar("LoadDomainType")
LoadValueType = TypeVar("LoadValueType")


class LoadCase(Generic[LoadDomainType, LoadValueType]):
    """
    Generic base class for all load cases.
    """

    def __init__(
        self,
        domain: LoadDomainType,
        value: LoadValueType,
    ):
        super().__init__()
        self._domain = domain
        self._value = value

    @property
    def domain(self) -> LoadDomainType:
        """Returns the domain of the load."""
        return self._domain

    @domain.setter
    def domain(self, value: LoadDomainType) -> None:
        """Sets the domain of the load."""
        self._domain = value

    @property
    def value(self) -> LoadValueType:
        """Returns the value of the load."""
        return self._value

    @value.setter
    def value(self, value: LoadValueType) -> None:
        """Sets the value of the load."""
        self._value = value

    @abstractmethod
    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        raise NotImplementedError("The method 'rhs' must be implemented.")


class RectangleLoad(LoadCase[Float2d, Float1d]):
    """
    A class to handle rectangular loads.

    Parameters
    ----------
    domain: :class:`~sigmaepsilon.solid.fourier.loads.Float2d`
       Load intensities for each dof in the order :math:`f_z, m_x, m_y`.
    value: :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
       The coordinates of the lower-left and upper-right points of the region
       where the load is applied. Default is ``None``.

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <plate_sign_conventions>` section of the theory guide.
    """

    @property
    def region(self) -> Iterable:
        """
        Returns the region as a list of 4 values x0, y0, w, and h, where x0 and y0 are
        the coordinates of the bottom-left corner, w and h are the width and height
        of the region.
        """
        return points_to_rectangle_region(np.array(self.domain))

    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the coefficients as a NumPy array.

        Parameters
        ----------
        problem: :class:`~sigmaepsilon.solid.fourier.problem.NavierProblem`
            A problem the coefficients are generated for. If not specified,
            the attached problem of the object is used. Default is None.

        Returns
        -------
        numpy.ndarray
            3d float array of shape (1, H, 3), where H is the total number
            of harmonic terms involved (defined for the problem). The first
            axis is always 1, as there is only one left hand side.
        """
        x = np.array(self.domain, dtype=float)
        v = np.array(self.value, dtype=float)
        return rhs_rect_const(problem.size, problem.shape, x, v)


class LineLoad(LoadCase[Float1d | Float2d, Float1d]):
    """
    A class to handle loads over lines for both beam and plate problems.

    Parameters
    ----------
    domain: :class:`~sigmaepsilon.solid.fourier.loads.Float1d` | :class:`~sigmaepsilon.solid.fourier.loads.Float2d`
        The point of application as an 1d iterable for a beam, a 2d iterable
        for a plate. In the latter case, the first row is the first point, the
        second row is the second point.
    value: :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        Load intensities for each dof. The order of the dofs for a beam
        is :math:`[F, M]`, for a plate it is :math:`[F, M_x, M_y]`.

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <sign_conventions>` section of the theory guide.
    """

    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the coefficients as a NumPy array.

        Parameters
        ----------
        problem: :class:`~sigmaepsilon.solid.fourier.problem.NavierProblem`
            A problem the coefficients are generated for. If not specified,
            the attached problem of the object is used. Default is None.

        Returns
        -------
        numpy.ndarray
            2d float array of shape (H, 3), where H is the total number
            of harmonic terms involved (defined for the problem).
        """
        x = np.array(self.domain, dtype=float)
        v = self.value
        if len(x.shape) == 1:
            assert len(v) == 2, f"Invalid shape {v.shape} for load intensities."
            if isinstance(v[0], Number) and isinstance(v[1], Number):
                v = np.array(v, dtype=float)
                return rhs_line_const(problem.length, problem.N, v, x)
            else:
                rhs = np.zeros((1, problem.N, 2), dtype=x.dtype)
                if isinstance(v[0], str):
                    f = Function(v[0], variables=["x"], dim=1)
                    L = problem.length
                    points = np.linspace(x[0], x[1], 1000)
                    d = x[1] - x[0]
                    rhs[0, :, 0] = list(
                        map(
                            lambda i: (2 / L)
                            * d
                            * avg(sin1d(points, i, L) * f([points])),
                            np.arange(1, problem.N + 1),
                        )
                    )
                elif isinstance(v[0], Number):
                    _v = np.array([v[0], 0], dtype=float)
                    rhs[0, :, 0] = rhs_line_const(problem.length, problem.N, _v, x)[
                        0, :, 0
                    ]
                if isinstance(v[1], str):
                    f = Function(v[1], variables=["x"], dim=1)
                    L = problem.length
                    points = np.linspace(x[0], x[1], 1000)
                    d = x[1] - x[0]
                    rhs[0, :, 1] = list(
                        map(
                            lambda i: (2 / L)
                            * d
                            * avg(cos1d(points, i, L) * f([points])),
                            np.arange(1, problem.N + 1),
                        )
                    )
                elif isinstance(v[1], Number):
                    _v = np.array([0, v[1]], dtype=float)
                    rhs[0, :, 1] = rhs_line_const(problem.length, problem.N, _v, x)[
                        0, :, 1
                    ]
                return rhs
        else:
            raise NotImplementedError(
                "Line loads are only implemented for 1d problems at the moment."
            )


class PointLoad(LoadCase[float | Float1d, Float1d]):
    """
    A class to handle concentrated loads.

    Parameters
    ----------
    domain: float or :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        The point of application. A scalar for a beam, an iterable of
        length 2 for a plate.
    value: :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        Load values for each dof. The order of the dofs for a beam
        is [F, M], for a plate it is [F, Mx, My].

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <sign_conventions>` section of the theory guide.
    """

    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the coefficients as a NumPy array.

        Parameters
        ----------
        problem: :class:`~sigmaepsilon.solid.fourier.problem.NavierProblem`
            A problem the coefficients are generated for. If not specified,
            the attached problem of the object is used. Default is ``None``.

        Returns
        -------
        numpy.ndarray
            2d float array of shape (H, 3), where H is the total number
            of harmonic terms involved (defined for the problem).
        """
        x = self.domain
        v = np.array(self.value)
        if hasattr(problem, "size"):
            return rhs_conc_2d(problem.size, problem.shape, v, x)
        else:
            return rhs_conc_1d(problem.length, problem.N, v, x)

class LoadGroup(DeepDict[Hashable, LoadGroupProtocol | LoadCaseProtocol | Any]):
    """
    A class to handle load groups for Navier's semi-analytic solution of
    rectangular plates and beams with specific boundary conditions.

    This class is also the base class of all other load types.

    See Also
    --------
    :class:`~sigmaepsilon.deepdict.deepdict.DeepDict`

    Examples
    --------
    >>> from sigmaepsilon.solid.fourier import LoadGroup, PointLoad
    >>>
    >>> loads = LoadGroup(
    >>>     group1 = LoadGroup(
    >>>         case1 = PointLoad(x=L/3, v=[1.0, 0.0]),
    >>>         case2 = PointLoad(x=L/3, v=[0.0, 1.0]),
    >>>     ),
    >>>     group2 = LoadGroup(
    >>>         case1 = PointLoad(x=2*L/3, v=[1.0, 0.0]),
    >>>         case2 = PointLoad(x=2*L/3, v=[0.0, 1.0]),
    >>>     ),
    >>> )

    Since the LoadGroup class is a subclass of :class:`~sigmaepsilon.deepdict.deepdict.DeepDict`,
    a case is accessible as

    >>> loads['group1', 'case1']

    If you want to protect the object from the accidental
    creation of nested subdirectories, you can lock the layout
    by typing

    >>> loads.lock()
    """

    def __init__(self, *args, cooperative: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._cooperative = cooperative

    @property
    def cooperative(self) -> bool:
        """
        Returns `True` if the load cases of this group can interact.
        """
        return self._cooperative

    @cooperative.setter
    def cooperative(self, value: bool):
        """
        Sets the cooperativity of the cases in the group.
        """
        self._cooperative = value

    def groups(
        self,
        *,
        inclusive: Optional[bool] = False,
        blocktype: Optional[Any] = None,
        deep: Optional[bool] = True,
    ) -> Iterable[LoadGroupProtocol]:
        """
        Returns a generator object that yields all the subgroups.

        Parameters
        ----------
        inclusive: bool, Optional
            If ``True``, returns the object the call is made upon.
            Default is False.
        blocktype: Any, Optional
            The type of the load groups to return. Default is ``None``, that
            returns all types.
        deep: bool, Optional
            If ``True``, all deep groups are returned separately. Default is ``True``.

        Yields
        ------
        :class:`~sigmaepsilon.solid.fourier.loads.LoadGroup`
        """
        dtype = LoadGroup if blocktype is None else blocktype
        return self.containers(inclusive=inclusive, dtype=dtype, deep=deep)

    def cases(
        self,
        case_type: Any = None,
    ) -> Iterable[LoadCaseProtocol]:
        """
        Returns a generator that yields the load cases in the group.

        Parameters
        ----------
        case_type: Any, Optional
            The type of the load cases to return. Default is None, that
            returns all types.

        Yields
        ------
        :class:`~sigmaepsilon.solid.fourier.loads.LoadCase`
        """
        case_type = LoadCase if case_type is None else case_type
        return filter(
            lambda i: isinstance(i, case_type),
            self.values(deep=True),
        )
