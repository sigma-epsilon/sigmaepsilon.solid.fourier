from typing import Iterable, ClassVar, TYPE_CHECKING, Sequence
import numpy as np


__all__ = [
    "LoadCaseResultLinStat",
    "BeamLoadCaseResultLinStat",
    "PlateLoadCaseResultLinStat",
]


if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr  # Only for type hints
    import pandas as pd  # Only for type hints


class LoadCaseResultLinStat:
    """
    A class to store results of linear static analysis for a single load case.
    """

    __slots__ = ["_data", "_name", "_components"]

    components: ClassVar[Iterable[str]]
    strain_range: ClassVar[Sequence[int]] = []

    def __init__(
        self,
        data,
        *,
        name: str | None = None,
    ):
        self._data = data
        self._name = name

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
    def strains(self) -> np.ndarray:
        """
        Returns the results as a ``numpy.ndarray``.
        """
        if self.strain_range is None:
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

    def to_xarray(self) -> "xr.DataArray":
        """
        Returns the results as an instance of  :class:`xarray.DataArray`.
        """
        import xarray as xr

        nP = len(self.data)
        components = self.components
        coords = [np.arange(nP), components]

        return xr.DataArray(
            self.data, coords=coords, dims=["index", "component"], name=self.name
        )

    def to_pandas(self) -> "pd.DataFrame":
        """
        Returns the results as an instance of :class:`pandas.DataFrame`.
        """
        import pandas as pd

        nP = len(self.data)
        components = self.components
        index = np.arange(nP)
        columns = components

        df = pd.DataFrame(self.data, index=index, columns=columns)
        df.index.name = "index"

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

    +-------+--------------------------------------------------+
    | Name  | Description                                      |
    +=======+==================================================+
    | UY    | Displacement in local Y direction                |
    +-------+--------------------------------------------------+
    | ROTZ  | Rotation around local Z axis                     |
    +-------+--------------------------------------------------+
    | CZ    | Curvature related to bending around local Z axis |
    +-------+--------------------------------------------------+
    | EXY   | Shear strain in local Y direction                |
    +-------+--------------------------------------------------+
    | MZ    | Bending moment around local Z axis               |
    +-------+--------------------------------------------------+
    | SY    | Shear force in local Y direction                 |
    +-------+--------------------------------------------------+

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <beam_sign_conventions>` section of the theory guide.

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


class PlateLoadCaseResultLinStat(LoadCaseResultLinStat):
    """
    A class to store results of linear static analysis for plates and
    a single load case.

    The class is a subclass of the base class :class:`~LoadCaseResultLinStat`, refer to its
    documentation for available methods and properties.

    The underlying data structure is a 2d NumPy array, where the first axis
    goes along the points of evaluation and the second axis goes along the
    following components:

    +-------+--------------------------------------------------+
    | Name  | Description                                      |
    +=======+==================================================+
    | UZ    | Displacement in local Z direction                |
    +-------+--------------------------------------------------+
    | ROTX  | Rotation around local X axis (CW)                |
    +-------+--------------------------------------------------+
    | ROTY  | Rotation around local Y axis (CW)                |
    +-------+--------------------------------------------------+
    | CX    | Curvature related to bending around local X axis |
    +-------+--------------------------------------------------+
    | CY    | Curvature related to bending around local Y axis |
    +-------+--------------------------------------------------+
    | CXY   | Twisting curvature                               |
    +-------+--------------------------------------------------+
    | EXZ   | Shear strain in local Y-Z plane                  |
    +-------+--------------------------------------------------+
    | EYZ   | Shear strain in local X-Z plane                  |
    +-------+--------------------------------------------------+
    | MX    | Bending moment around local Y axis (CCW)         |
    +-------+--------------------------------------------------+
    | MY    | Bending moment around local X axis (CW)          |
    +-------+--------------------------------------------------+
    | MXY   | Twisting moment around local Z axis (CW)         |
    +-------+--------------------------------------------------+
    | QX    | Shear force on the local Y-Z plane (+Z)          |
    +-------+--------------------------------------------------+
    | QY    | Shear force on the local X-Z plane  (+Z)         |
    +-------+--------------------------------------------------+

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <plate_sign_conventions>` section of the theory guide.

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
    strain_range = range(3, 8)
    
    @property
    def strains(self) -> np.ndarray:
        """
        Returns the results as a ``numpy.ndarray``.
        """
        return self._data[:, 3:8]
        return self._data[:, list(self.strain_range)]
