from typing import Iterable

import numpy as np
from numpy import ndarray

from ..preproc import rhs_rect_const
from ..utils import points_to_rectangle_region
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d, Float2d

__all__ = ["RectangleLoad"]


class RectangleLoad(LoadCase[Float2d, Float1d]):
    """
    A class to handle loads defined over a single rectangle.

    Parameters
    ----------
    domain: :class:`~sigmaepsilon.solid.fourier.loads.Float2d`
        The coordinates of the lower-left and upper-right points of the region
        where the load is applied. Default is ``None``.
    value: :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        Load intensities for each dof in the order :math:`f_z, m_x, m_y`.
       
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
