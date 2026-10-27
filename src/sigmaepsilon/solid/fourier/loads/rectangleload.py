from collections.abc import Iterable

import numpy as np
from numpy import ndarray

from ..integration import integrate_rectangle
from ..preproc import rhs0_rect_const, rhs_rect_const
from ..protocols import NavierProblemProtocol
from ..utils import check_points_in_domain, points_to_rectangle_region
from .loads import Float1d, Float2d, LoadCase

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
    num_mc: int, Optional
        The number of sampling points for Monte Carlo integration. If no value
        is provided, the global config value is used.
    integration: str, Optional
        The method of numerical integration, see
        :class:`~sigmaepsilon.solid.fourier.loads.LoadCase`.
    rng: int or numpy.random.Generator, Optional
        The seed or the random generator of Monte Carlo integration, see
        :class:`~sigmaepsilon.solid.fourier.loads.LoadCase`.

    Note
    ----
    The coefficients of constant loads are calculated in closed form, the ones
    of loads given as string expressions are integrated numerically.


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
        return self._rhs(problem, zero_order=False)

    def rhs0(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the zero-order coefficients of the moment loads. See
        :func:`~sigmaepsilon.solid.fourier.loads.LoadCase.rhs0` for details.
        """
        return self._rhs(problem, zero_order=True)

    def _rhs(self, problem: NavierProblemProtocol, zero_order: bool) -> ndarray:
        assert problem.model_type.is_2d, f"Invalid model type {problem.model_type}."

        domain = np.array(self.domain, dtype=float)
        values = self.value
        check_points_in_domain(domain, problem.size, "corners of the rectangle")

        has_symbolic_load = any(not isinstance(vi, (float, int)) for vi in values)
        if has_symbolic_load:
            return integrate_rectangle(
                problem.size,
                problem.shape,
                domain,
                values,
                zero_order=zero_order,
                **self._integration_options(),
            )
        else:
            values = np.array(values, dtype=float)
            evaluator = rhs0_rect_const if zero_order else rhs_rect_const
            return evaluator(problem.size, problem.shape, domain, values)
