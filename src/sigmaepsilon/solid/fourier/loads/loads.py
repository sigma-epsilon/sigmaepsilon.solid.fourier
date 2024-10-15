from typing import Iterable, Any, Optional, TypeVar, Generic, TypeAlias, Hashable
from types import NoneType
from abc import abstractmethod

from numpy import ndarray
import numpy as np

from sigmaepsilon.deepdict import DeepDict

from ..protocols import NavierProblemProtocol, LoadGroupProtocol, LoadCaseProtocol
from ..postproc import eval_loads_2d
from ..config import __hasmatplotlib__

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

    def _eval_at_points(self, points: Iterable) -> ndarray:  # pragma: no cover
        """
        Evaluates the load function at the given points.
        """
        raise NotImplementedError

    def _gen_plot_points(
        self, problem: NavierProblemProtocol, grid_shape: tuple[int, int]
    ) -> ndarray:
        """
        Generates a collection of points for plotting. Sometimes it is useful
        to override this method to have a more meaningful output. For instance,
        for a point load, it is quite important that there is a point at the
        location of the load.
        """
        length_X, length_Y = problem.size
        n_X, n_Y = grid_shape
        x = np.linspace(0, length_X, n_X)
        y = np.linspace(0, length_Y, n_Y)
        xv, yv = np.meshgrid(x, y)
        points = np.stack((xv.flatten(), yv.flatten()), axis=1)
        return points

    @abstractmethod
    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        raise NotImplementedError("The method 'rhs' must be implemented.")

    def plot_mpl_3d(
        self,
        problem: NavierProblemProtocol,
        *,
        ax=None,
        points: Iterable | NoneType = None,
        grid_shape: tuple[int, int] | NoneType = None,
        return_axis: bool = False,
        **plot_kwargs,
    ) -> object:
        """
        Plots the load on the provided axis using the `plot_surface`
        method of `matplotlib`.

        Parameters
        ----------
        problem: :class:`~sigmaepsilon.solid.fourier.problem.NavierProblem`
            A problem the coefficients are generated for.
        ax: matplotlib.axes.Axes
            The axis to plot on. If not specified, a new axis is created.
        **plot_kwargs
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object.

        """
        if not __hasmatplotlib__:  # pragma: no cover
            raise ImportError("matplotlib is not available.")

        import matplotlib.pyplot as plt
        from matplotlib import cm

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        if grid_shape is None:
            grid_shape = problem.shape

        if points is None:
            points = self._gen_plot_points(problem, grid_shape)
        points = np.array(points).astype(float)

        rhs = self.rhs(problem)
        load_values = self._eval_at_points(points)

        length_X, length_Y = problem.size
        number_of_modes_X, number_of_modes_Y = problem.shape

        values = eval_loads_2d(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            rhs,
            points,
            load_values,
        )

        X = points[:, 0].reshape((number_of_modes_X, number_of_modes_Y))
        Y = points[:, 1].reshape((number_of_modes_X, number_of_modes_Y))
        Z = values[0, :, 0].reshape((number_of_modes_X, number_of_modes_Y))

        if "cmap" not in plot_kwargs:
            plot_kwargs["cmap"] = cm.turbo

        if "linewidth" not in plot_kwargs:
            plot_kwargs["linewidth"] = 0

        if "antialiased" not in plot_kwargs:
            plot_kwargs["antialiased"] = True

        ax.plot_surface(X, Y, Z, **plot_kwargs)

        if return_axis:
            return ax


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
