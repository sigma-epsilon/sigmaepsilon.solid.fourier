import numpy as np

from ..preproc import rhs_line_1d_any, rhs_line_2d_any
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d, Float2d


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

    .. hint::
        For a detailed explanation of the sign conventions, refer to
        :ref:`this <sign_conventions>` section of the theory guide.
        
    """

    def rhs(self, problem: NavierProblemProtocol) -> np.ndarray:
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
        v = self.value
        if len(x.shape) == 1:
            assert len(v) == 2, f"Invalid shape {v.shape} for load intensities."
            return rhs_line_1d_any(problem.length, problem.N, v, x)
        elif len(x.shape) == 2:
            assert len(v) == 3, f"Invalid shape {v.shape} for load intensities."
            return rhs_line_2d_any(problem.size, problem.shape, v, x)
        else:  # pragma: no cover
            raise ValueError("Invalid shape for the domain.")
