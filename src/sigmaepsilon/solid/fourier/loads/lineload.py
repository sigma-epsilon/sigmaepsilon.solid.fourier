from functools import partial

import numpy as np
from numpy import ndarray

from ..preproc import rhs_line_const, rhs_line_1d_mc, rhs_line_2d_mc
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d, Float2d
from ..utils import is_scalar
from ..config import Config

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
        domain = np.array(self.domain, dtype=float)
        values = self.value
        n_MC = self._num_mc or Config.NUM_MC_SAMPLES_PLATE

        if problem.model_type.is_2d:
            assert (
                len(domain.shape) == 2
            ), f"Invalid shape {domain.shape} for the domain."
            assert len(values) == 3, "Invalid shape for load intensities."

            evaluator = partial(rhs_line_2d_mc, n_MC=n_MC)
        elif problem.model_type.is_1d:
            assert (
                len(domain.shape) == 1
            ), f"Invalid shape {domain.shape} for the domain."
            assert (
                len(values) == 2
            ), f"Invalid shape {values.shape} for load intensities."

            if is_scalar(values[0]) and is_scalar(values[1]):
                values = np.array(values, dtype=float)
                evaluator = rhs_line_const
            else:
                evaluator = partial(rhs_line_1d_mc, n_MC=n_MC)
        else:  # pragma: no cover
            raise NotImplementedError

        return evaluator(problem.size, problem.shape, domain, values)
