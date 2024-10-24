from typing import Iterable, Any, Optional, TypeVar, Generic, TypeAlias, Hashable
from types import NoneType
from abc import abstractmethod

from numpy import ndarray
import numpy as np

from sigmaepsilon.deepdict import DeepDict

from ..protocols import NavierProblemProtocol, LoadGroupProtocol, LoadCaseProtocol
from ..postproc import eval_loads_1d, eval_loads_2d
from ..enums import MechanicalModelType
from ..config import Config

__all__ = ["LoadGroup", "LoadCase", "RectangleLoad", "LineLoad", "PointLoad"]


Float1d: TypeAlias = Iterable[float]
Float2d: TypeAlias = Iterable[Float1d]

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
        The number of sampling points for Monte Carlo integration. 
        If no value is provided, the global config value is used.
        
    """

    def __init__(
        self,
        domain: LoadDomainType,
        value: LoadValueType,
        num_mc: int | NoneType = None,
    ):
        super().__init__()
        self._domain = domain
        self._value = value
        self._num_mc = num_mc

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

    def eval_approx(self, problem: NavierProblemProtocol, points: Iterable) -> ndarray:
        """
        Evaluates the Fourier series approximation of the load at the given points.

        The returned array is an 1d array with the same length as the number of points.
        """
        if problem.model_type in [
            MechanicalModelType.KIRCHHOFF_LOVE_PLATE,
            MechanicalModelType.UFLYAND_MINDLIN_PLATE,
        ]:
            length_X, length_Y = problem.size
            number_of_modes_X, number_of_modes_Y = problem.shape
            rhs = self.rhs(problem)
            points = np.array(points, dtype=float)
            return eval_loads_2d(
                (length_X, length_Y),
                (number_of_modes_X, number_of_modes_Y),
                rhs,
                points,
            )
        elif problem.model_type in [
            MechanicalModelType.BERNOULLI_EULER_BEAM,
            MechanicalModelType.TIMOSHENKO_BEAM,
        ]:
            length_X = float(problem.size)
            number_of_modes_X = problem.shape
            rhs = self.rhs(problem)
            points = np.array(points, dtype=float)
            return eval_loads_1d(
                length_X,
                number_of_modes_X,
                rhs,
                points,
            )
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
