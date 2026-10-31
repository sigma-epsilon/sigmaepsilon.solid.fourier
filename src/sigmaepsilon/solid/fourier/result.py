from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np

__all__ = [
    "LoadCaseResultLinStat",
    "BeamLoadCaseResultLinStat",
    "PlateLoadCaseResultLinStat",
]


if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # Only for type hints
    import xarray as xr  # Only for type hints


class LoadCaseResultLinStat:
    """
    A class to store results of a linear static analysis for a single load case.

    Results returned by a linear static analysis also know the points they were
    evaluated at (see :attr:`points`) and are able to evaluate the solution of
    their load case at further points (see :func:`at`).
    """

    __slots__ = ["_components", "_data", "_evaluator", "_name", "_points"]

    components: ClassVar[Iterable[str]]
    coordinates: ClassVar[Sequence[str]] = ()
    strain_range: ClassVar[Sequence[int]] = []

    def __init__(
        self,
        data,
        *,
        name: str | None = None,
        points: Iterable | None = None,
        evaluator: Callable | None = None,
    ):
        self._data = data
        self._name = name
        if points is not None:
            points = np.asarray(points, dtype=float).reshape(len(data), -1)
        self._points = points
        self._evaluator = evaluator

    def __getitem__(self, key: str | Sequence[str]) -> np.ndarray:
        """
        Returns one or more components of the results by their names, as an 1d
        array for a single name and as a 2d array for a sequence of names.
        """
        components = list(self.components)
        if isinstance(key, str):
            return self._data[:, self._component_index(components, key)]
        indices = [self._component_index(components, k) for k in key]
        return self._data[:, indices]

    @staticmethod
    def _component_index(components: list[str], key: str) -> int:
        try:
            return components.index(key)
        except ValueError:
            raise KeyError(
                f"Unknown component {key!r}, the available ones are {components}."
            ) from None

    @property
    def data(self) -> np.ndarray:
        """
        Returns the results as a ``numpy.ndarray``.
        """
        return self._data

    @property
    def values(self) -> np.ndarray:
        """
        Returns the results as a ``numpy.ndarray``.
        """
        return self._data

    @property
    def points(self) -> np.ndarray | None:
        """
        Returns the points of evaluation as a 2d array of shape (nP, nD), where nD
        is 1 for a beam and 2 for a plate, or `None` if they are not known.
        """
        return self._points

    @property
    def strains(self) -> np.ndarray:
        """
        Returns the strains as a ``numpy.ndarray``.
        """
        if self.strain_range is None:  # pragma: no cover
            raise ValueError("Strain range is not defined.")
        return self._data[:, list(self.strain_range)]

    @property
    def name(self) -> str | None:
        """
        Returns the name of the results.
        """
        return self._name

    @name.setter
    def name(self, value: str | None):
        """
        Sets the name of the results.
        """
        self._name = value

    def at(self, points: float | Iterable) -> "LoadCaseResultLinStat":
        """
        Evaluates the solution of the load case at one or more points and returns
        the results as a new instance of the same class. The values are exact at
        any point of the domain, since the solution is a Fourier series and there
        is no interpolation involved.

        Parameters
        ----------
        points: float or Iterable
            For a beam, a scalar or an 1d iterable of coordinates. For a plate,
            a single point as ``(x, y)`` or a 2d iterable of shape (nP, 2).

        Example
        -------
        >>> result = solution["LG1", "LC1"]  # doctest: +SKIP
        >>> result.at([(100.0, 200.0), (300.0, 400.0)])["UZ"]  # doctest: +SKIP
        """
        if self._evaluator is None:
            raise ValueError(
                "These results can not be evaluated at new points, only the ones "
                "returned by a linear static analysis can."
            )
        points, data = self._evaluator(points)
        return type(self)(
            data, name=self.name, points=points, evaluator=self._evaluator
        )

    def _coordinate_columns(self) -> dict[str, np.ndarray]:
        if self._points is None:
            return {}
        return {c: self._points[:, i] for i, c in enumerate(self.coordinates)}

    def to_xarray(self) -> "xr.DataArray":
        """
        Returns the results as an instance of  :class:`xarray.DataArray`.

        If the points of evaluation are known, their coordinates are attached
        to the ``index`` dimension as coordinates named after the axes
        (``x`` for beams, ``x`` and ``y`` for plates).
        """
        import xarray as xr

        nP = len(self.data)
        coords = {"index": np.arange(nP), "component": list(self.components)}
        for c, values in self._coordinate_columns().items():
            coords[c] = ("index", values)

        return xr.DataArray(
            self.data, coords=coords, dims=["index", "component"], name=self.name
        )

    def to_pandas(self, coords: bool = False) -> "pd.DataFrame":
        """
        Returns the results as an instance of :class:`pandas.DataFrame`.

        Parameters
        ----------
        coords: bool, Optional
            If `True` and the points of evaluation are known, their coordinates
            are added in front of the components as columns named after the axes
            (``x`` for beams, ``x`` and ``y`` for plates). Default is `False`.
        """
        import pandas as pd

        nP = len(self.data)
        components = self.components
        index = np.arange(nP)
        columns = components

        df = pd.DataFrame(self.data, index=index, columns=columns)
        df.index.name = "index"

        if coords:
            for i, (c, values) in enumerate(self._coordinate_columns().items()):
                df.insert(i, c, values)

        return df


class BeamLoadCaseResultLinStat(LoadCaseResultLinStat):
    """
    A class to store results of linear static analysis for beams and
    a single load case.

    The class is a subclass of the base class :class:`~LoadCaseResultLinStat`, refer to its
    documentation for available methods and properties.

    The underlying data structure is a 2d NumPy array, where the first axis
    goes along the points of evaluation and the second axis goes along the
    following components:

    .. hint::
        For a detailed explanation of the sign conventions, refer to the
        :ref:`theory guide <theory_guide>`.

    +-------+-------+--------------------------------------------------+
    | Index | Name  | Description                                      |
    +=======+=======+==================================================+
    | 0     | UY    | Displacement in Y direction                      |
    +-------+-------+--------------------------------------------------+
    | 1     | ROTZ  | Rotation around Z axis                           |
    +-------+-------+--------------------------------------------------+
    | 2     | CZ    | Curvature related to bending around Z axis       |
    +-------+-------+--------------------------------------------------+
    | 3     | EXY   | Engineering shear strain in the X-Y plane        |
    +-------+-------+--------------------------------------------------+
    | 4     | MZ    | Bending moment around Z axis                     |
    +-------+-------+--------------------------------------------------+
    | 5     | SY    | Shear force in Y direction                       |
    +-------+-------+--------------------------------------------------+

    See also
    --------
    :class:`~LoadCaseResultLinStat`

    """

    components = [
        "UY",
        "ROTZ",
        "CZ",
        "EXY",
        "MZ",
        "SY",
    ]
    coordinates = ("x",)
    strain_range = range(2, 4)

    def to_xarray(self) -> "xr.DataArray":
        """
        Returns the results as an instance of  :class:`xarray.DataArray`.

        If the points of evaluation are known and strictly monotonic, the ``x``
        coordinates are indexed, so that results can be selected by location::

            result.to_xarray().sel(x=1.5, method="nearest").sel(component="UY")

        Otherwise, the ``x`` coordinates are attached without an index.
        """
        xarr = super().to_xarray()
        if self._points is not None and len(self._points) > 0:
            x = self._points[:, 0]
            dx = np.diff(x)
            if np.all(dx > 0) or np.all(dx < 0):
                xarr = xarr.set_xindex("x")
        return xarr


class PlateLoadCaseResultLinStat(LoadCaseResultLinStat):
    """
    A class to store results of linear static analysis for plates and
    a single load case.

    The class is a subclass of the base class :class:`~LoadCaseResultLinStat`, refer to its
    documentation for available methods and properties.

    The underlying data structure is a 2d NumPy array, where the first axis
    goes along the points of evaluation and the second axis goes along the
    following components:

    .. hint::
        For a detailed explanation of the sign conventions, refer to the
        :ref:`theory guide <theory_guide>`.

    +-------+-------+--------------------------------------------------+
    | Index | Name  | Description                                      |
    +=======+=======+==================================================+
    | 0     | UZ    | Displacement in Z direction                      |
    +-------+-------+--------------------------------------------------+
    | 1     | ROTX  | Rotation around X axis (CCW)                     |
    +-------+-------+--------------------------------------------------+
    | 2     | ROTY  | Rotation around Y axis (CCW)                     |
    +-------+-------+--------------------------------------------------+
    | 3     | CX    | Curvature related to bending around X axis       |
    +-------+-------+--------------------------------------------------+
    | 4     | CY    | Curvature related to bending around Y axis       |
    +-------+-------+--------------------------------------------------+
    | 5     | CXY   | Twisting curvature                               |
    +-------+-------+--------------------------------------------------+
    | 6     | EXZ   | Engineering shear strain in X-Z plane            |
    +-------+-------+--------------------------------------------------+
    | 7     | EYZ   | Engineering shear strain in Y-Z plane            |
    +-------+-------+--------------------------------------------------+
    | 8     | MX    | Bending moment around Y axis (CCW)               |
    +-------+-------+--------------------------------------------------+
    | 9     | MY    | Bending moment around X axis (CW)                |
    +-------+-------+--------------------------------------------------+
    | 10    | MXY   | Twisting moment                                  |
    +-------+-------+--------------------------------------------------+
    | 11    | QX    | Shear force on the X-Z plane (+Z)                |
    +-------+-------+--------------------------------------------------+
    | 12    | QY    | Shear force on the Y-Z plane  (+Z)               |
    +-------+-------+--------------------------------------------------+

    See also
    --------
    :class:`~LoadCaseResultLinStat`

    """

    components = [
        "UZ",
        "ROTX",
        "ROTY",
        "CX",
        "CY",
        "CXY",
        "EXZ",
        "EYZ",
        "MX",
        "MY",
        "MXY",
        "QX",
        "QY",
    ]
    coordinates = ("x", "y")
    strain_range = range(3, 8)

    def to_xarray(self, grid: bool = False) -> "xr.DataArray":
        """
        Returns the results as an instance of  :class:`xarray.DataArray`.

        Parameters
        ----------
        grid: bool, Optional
            If `False`, the array has the dimensions ``index`` and ``component``,
            and the coordinates of the points are attached to the ``index``
            dimension as ``x`` and ``y``. If `True`, the points must form a full
            rectangular grid (every combination of the distinct x and y
            coordinates, in any order), and the array has the dimensions ``x``,
            ``y`` and ``component``, so that results can be selected by location::

                xarr = result.to_xarray(grid=True)
                xarr.sel(x=100.0, y=200.0, method="nearest").sel(component="UZ")

            Default is `False`.
        """
        xarr = super().to_xarray()
        if not grid:
            return xarr

        if self._points is None:
            raise ValueError("The points of evaluation are not known.")
        x, ix = _unique_with_tolerance(self._points[:, 0])
        y, iy = _unique_with_tolerance(self._points[:, 1])
        flat = ix * len(y) + iy
        if len(x) * len(y) != len(flat) or len(np.unique(flat)) != len(flat):
            raise ValueError(
                "The points of evaluation do not form a full rectangular grid."
            )
        # the coordinates are replaced by the distinct values they are grouped by
        xarr = xarr.assign_coords(x=("index", x[ix]), y=("index", y[iy]))
        xarr = xarr.set_index(index=["x", "y"]).unstack("index")
        return xarr.transpose("x", "y", "component")


def _unique_with_tolerance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the sorted distinct values of an 1d array and the position of each
    item among them, where values closer than a small relative tolerance are
    treated as equal.
    """
    if len(values) == 0:
        return values, np.zeros(0, dtype=int)
    order = np.argsort(values)
    sorted_values = values[order]
    tol = 1e-9 * max(np.ptp(values), 1.0)
    is_new = np.concatenate([[True], np.diff(sorted_values) > tol])
    labels = np.empty(len(values), dtype=int)
    labels[order] = np.cumsum(is_new) - 1
    return sorted_values[is_new], labels
