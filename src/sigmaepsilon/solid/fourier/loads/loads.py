from abc import abstractmethod
from collections.abc import Hashable, Iterable
from types import NoneType
from typing import Any, Generic, Optional, TypeAlias, TypeVar

import numpy as np
from numpy import ndarray
from sigmaepsilon.deepdict import DeepDict

from ..config import Config
from ..enums import IntegrationMethod
from ..postproc import eval_loads_1d, eval_loads_2d
from ..protocols import LoadCaseProtocol, LoadGroupProtocol, NavierProblemProtocol
from ..utils import RNGType, to_positive_int

__all__ = ["LoadCase", "LoadGroup"]


Float1d: TypeAlias = Iterable[float]
Float2d: TypeAlias = Iterable[Float1d]

Int1d: TypeAlias = Iterable[int]
Int2d: TypeAlias = Iterable[Int1d]

LoadDomainType = TypeVar("LoadDomainType")
LoadValueType = TypeVar("LoadValueType")


class LoadCase(Generic[LoadDomainType, LoadValueType]):
    """
    Generic base class for all load cases.

    Parameters
    ----------
    domain: LoadDomainType
        The domain of the load.
    value: LoadValueType
        The value of the load.
    num_mc: int, Optional
        The number of sampling points for Monte Carlo integration. If no value
        is provided, the global config value is used. Providing it selects Monte
        Carlo integration, unless `integration` says otherwise.
    integration: str or :class:`~sigmaepsilon.solid.fourier.enums.IntegrationMethod`, Optional
        The method of numerical integration of the load coefficients, either
        ``"quadrature"`` (Gauss-Legendre) or ``"mc"`` (Monte Carlo). It is used
        for loads whose coefficients have no closed form, for instance if any of
        the values is a string expression. If not provided, it is ``"mc"`` if
        `num_mc` is provided, and `Config.INTEGRATION_METHOD` otherwise.
    rng: int or numpy.random.Generator, Optional
        The seed or the random generator of Monte Carlo integration, see
        :func:`numpy.random.default_rng`. With a seed, the coefficients are the
        same every time they are calculated. If not provided, `Config.MC_SEED`
        is used. The global random state is never used.

    """

    def __init__(
        self,
        domain: LoadDomainType,
        value: LoadValueType,
        num_mc: int | NoneType = None,
        *,
        integration: str | IntegrationMethod | NoneType = None,
        rng: RNGType = None,
    ):
        super().__init__()
        self._domain = domain
        self._value = value
        self._num_mc = None if num_mc is None else to_positive_int(num_mc, "num_mc")
        self._integration = (
            None if integration is None else IntegrationMethod.from_value(integration)
        )
        self._rng = rng

    @property
    def integration(self) -> IntegrationMethod:
        """
        Returns the method of numerical integration of the load coefficients.
        """
        if self._integration is not None:
            return self._integration
        if self._num_mc is not None:
            return IntegrationMethod.MONTE_CARLO
        return Config.INTEGRATION_METHOD

    @integration.setter
    def integration(self, value: str | IntegrationMethod | NoneType) -> None:
        """
        Sets the method of numerical integration of the load coefficients.
        """
        self._integration = (
            None if value is None else IntegrationMethod.from_value(value)
        )

    def _integration_options(self) -> dict:
        rng = Config.MC_SEED if self._rng is None else self._rng
        return {"method": self.integration, "num_mc": self._num_mc, "rng": rng}

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

    @abstractmethod
    def rhs0(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the zero-order coefficients of the moment loads. Moment loads are
        expanded in series that are cosines in one direction, and the terms with a
        zero index in that direction are not covered by :func:`rhs`.

        For a beam, the result is an 1d array of shape (2,), where the second item
        is the mean of the moment load over the span. For a plate, it is a 2d array
        of shape (M + N, 3), where the first M rows hold the coefficients of
        :math:`m_x` for the harmonics :math:`(m, 0)` and the next N rows hold the
        coefficients of :math:`m_y` for the harmonics :math:`(0, n)`.
        """
        raise NotImplementedError("The method 'rhs0' must be implemented.")

    def eval_approx(self, problem: NavierProblemProtocol, points: Iterable) -> ndarray:
        """
        Evaluates the Fourier series approximation of the load at the given points,
        including the zero-order terms of the moment loads (see :func:`rhs0`).

        The returned array is a 2d array of shape (nP, nDOF), where nP is the number
        of points and nDOF is the number of load components (2 for a beam, 3 for a
        plate).
        """
        if problem.model_type.is_2d:
            length_X, length_Y = problem.size
            number_of_modes_X, number_of_modes_Y = problem.shape
            rhs = self.rhs(problem)
            rhs0 = self.rhs0(problem)
            points = np.array(points, dtype=float)
            res = eval_loads_2d(
                (length_X, length_Y),
                (number_of_modes_X, number_of_modes_Y),
                rhs,
                points,
            )
            # mx at the harmonics (m, 0) and my at the harmonics (0, n)
            m = np.arange(1, number_of_modes_X + 1)
            n = np.arange(1, number_of_modes_Y + 1)
            Sm = np.sin(np.outer(points[:, 0], m) * np.pi / length_X)
            Sn = np.sin(np.outer(points[:, 1], n) * np.pi / length_Y)
            res[:, 1] += Sm @ rhs0[:number_of_modes_X, 1]
            res[:, 2] += Sn @ rhs0[number_of_modes_X:, 2]
            return res
        elif problem.model_type.is_1d:
            length_X = float(problem.size)
            number_of_modes_X = problem.shape
            rhs = self.rhs(problem)
            points = np.array(points, dtype=float)
            res = eval_loads_1d(
                length_X,
                number_of_modes_X,
                rhs,
                points,
            )
            # the mean of the moment load
            res[:, 1] += self.rhs0(problem)[1]
            return res
        else:  # pragma: no cover
            raise NotImplementedError


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
        *,
        return_address: bool = False,
    ) -> Iterable[LoadCaseProtocol] | Iterable[tuple[list, LoadCaseProtocol]]:
        """
        Returns a generator that yields the load cases in the group.

        Parameters
        ----------
        case_type: Any, Optional
            The type of the load cases to return. Default is None, that
            returns all types.
        return_address: bool, Optional
            If ``True``, the generator yields ``(address, case)`` pairs, where
            the address is the position of the case in the group.
            Default is ``False``.

        Yields
        ------
        :class:`~sigmaepsilon.solid.fourier.loads.LoadCase`
        """
        case_type = LoadCase if case_type is None else case_type
        items = self.items(deep=True, return_address=True, vtype=case_type)
        if return_address:
            return items
        return (case for _, case in items)
