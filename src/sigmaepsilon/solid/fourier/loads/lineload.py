import numpy as np
from numpy import ndarray

from ..preproc import rhs_line_const, rhs0_line_const
from ..integration import integrate_line_1d, integrate_line_2d
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d, Float2d
from ..utils import is_scalar, check_points_in_domain

__all__ = ["LineLoad"]


class LineLoad(LoadCase[Float1d | Float2d, Float1d]):
    """
    A class to handle loads over lines for both beam and plate problems.

    Parameters
    ----------
    domain: :class:`~sigmaepsilon.solid.fourier.loads.Float1d` | :class:`~sigmaepsilon.solid.fourier.loads.Float2d`
        The point of application as an 1d iterable for a beam, a 2d iterable
        for a plate. In the latter case, the first row is the first point, the
        second row is the second point.
    value: :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        Load intensities for each dof. The order of the dofs for a beam
        is :math:`[F, M]`, for a plate it is :math:`[F, M_x, M_y]`.
    num_mc: int, Optional
        The number of sampling points for Monte Carlo integration.
        If no value is provided, the global config value is used.
    integration: str, Optional
        The method of numerical integration, see
        :class:`~sigmaepsilon.solid.fourier.loads.LoadCase`.
    rng: int or numpy.random.Generator, Optional
        The seed or the random generator of Monte Carlo integration, see
        :class:`~sigmaepsilon.solid.fourier.loads.LoadCase`.

    Note
    ----
    On a beam, the coefficients of constant loads are calculated in closed
    form. On a plate, and for loads given as string expressions, they are
    integrated numerically.


    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <sign_conventions>` section of the theory guide.

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
        domain = np.array(self.domain, dtype=float)
        values = self.value

        if problem.model_type.is_2d:
            assert (
                len(domain.shape) == 2
            ), f"Invalid shape {domain.shape} for the domain."
            assert len(values) == 3, "Invalid shape for load intensities."
            check_points_in_domain(domain, problem.size, "end points of the line")

            return integrate_line_2d(
                problem.size,
                problem.shape,
                domain,
                values,
                zero_order=zero_order,
                **self._integration_options(),
            )
        elif problem.model_type.is_1d:
            assert (
                len(domain.shape) == 1
            ), f"Invalid shape {domain.shape} for the domain."
            assert (
                len(values) == 2
            ), f"Invalid shape {values.shape} for load intensities."
            check_points_in_domain(domain, problem.size, "end points of the line")
            if domain[0] > domain[1]:
                raise ValueError(
                    f"The start of the line must not be after its end, got {domain}."
                )

            if is_scalar(values[0]) and is_scalar(values[1]):
                values = np.array(values, dtype=float)
                evaluator = rhs0_line_const if zero_order else rhs_line_const
                return evaluator(problem.size, problem.shape, domain, values)

            return integrate_line_1d(
                problem.size,
                problem.shape,
                domain,
                values,
                zero_order=zero_order,
                **self._integration_options(),
            )
        else:  # pragma: no cover
            raise NotImplementedError
