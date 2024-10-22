from typing import Iterable

import numpy as np
from numpy import ndarray

from ..preproc import rhs_rect_const, rhs_rect_mc
from ..utils import points_to_rectangle_region
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d, Float2d
from ..enums import MechanicalModelType

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
            2d float array of shape (H, 3), where H is the total number
            of harmonic terms involved (defined for the problem).
        """
        assert problem.model_type in [
            MechanicalModelType.KIRCHHOFF_LOVE_PLATE,
            MechanicalModelType.UFLYAND_MINDLIN_PLATE,
        ], f"Invalid model type {problem.model_type}."

        domain = np.array(self.domain, dtype=float)
        values = self.value

        has_symbolic_load = any(not isinstance(vi, (float, int)) for vi in values)
        if has_symbolic_load:
            return rhs_rect_mc(problem.size, problem.shape, domain, values)
        else:
            values = np.array(values, dtype=float)
            return rhs_rect_const(problem.size, problem.shape, domain, values)
