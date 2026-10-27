from collections.abc import Iterable

import numpy as np
from numpy import ndarray

from ..integration import integrate_disk
from ..protocols import NavierProblemProtocol
from ..utils import check_points_in_domain, to_positive_float
from .loads import LoadCase

__all__ = ["DiskLoad"]


class DiskLoad(LoadCase[tuple[tuple[float, float], float], Iterable]):
    """
    A class to handle loads defined over a single disk.

    Parameters
    ----------
    domain: tuple[tuple[float, float], float]
        The first value is the coordinates of the center of the disk, the
        second one is the radius.
    value: Iterable
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
    The coefficients are always integrated numerically, and the disk must lie
    within the plate.

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <plate_sign_conventions>` section of the theory guide.

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
        return self._rhs(problem, zero_order=False)

    def rhs0(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the zero-order coefficients of the moment loads.

        See :func:`~sigmaepsilon.solid.fourier.loads.LoadCase.rhs0` for details.
        """
        return self._rhs(problem, zero_order=True)

    def _rhs(self, problem: NavierProblemProtocol, zero_order: bool) -> ndarray:
        assert problem.model_type.is_2d, f"Invalid model type {problem.model_type}."
        center, radius = self.domain
        center = np.array(center, dtype=float)
        radius = to_positive_float(radius, "radius")
        corners = [center - radius, center + radius]
        check_points_in_domain(corners, problem.size, "disk")
        return integrate_disk(
            problem.size,
            problem.shape,
            (center, radius),
            self.value,
            zero_order=zero_order,
            **self._integration_options(),
        )
