from os.path import dirname, abspath
from importlib.metadata import metadata

from sigmaepsilon.core.config import namespace_package_name

from .beam import NavierBeam
from .plate import RectangularPlate
from .loads import LoadGroup, RectangleLoad, LineLoad, PointLoad
from .exceptions import NavierLoadError

__all__ = [
    "NavierBeam",
    "RectangularPlate",
    "NavierLoadError",
    "LoadGroup",
    "RectangleLoad",
    "LineLoad",
    "PointLoad",
]

# __pkg_name__ = namespace_package_name(dirname(abspath(__file__)), 10)
__pkg_name__ = "sigmaepsilon.solid.fourier"
__pkg_metadata__ = metadata(__pkg_name__)
__version__ = __pkg_metadata__["version"]
__description__ = __pkg_metadata__["summary"]
del __pkg_metadata__
