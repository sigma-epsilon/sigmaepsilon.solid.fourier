import numpy as np
from numpy import ndarray

from ..preproc import rhs_line_const, rhs_line_1d_mc, rhs_line_2d_mc
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d, Float2d
from ..enums import MechanicalModelType

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
            axis is always of length 1, as there is only one left hand side.
        """
        x = np.array(self.domain, dtype=float)
        v = self.value

        if problem.model_type in [
            MechanicalModelType.KIRCHHOFF_LOVE_PLATE,
            MechanicalModelType.UFLYAND_MINDLIN_PLATE,
        ]:
            assert len(x.shape) == 2, f"Invalid shape {x.shape} for the domain."
            assert len(v) == 3, f"Invalid shape {v.shape} for load intensities."
            evaluator = rhs_line_2d_mc
        elif problem.model_type in [
            MechanicalModelType.BERNOULLI_EULER_BEAM,
            MechanicalModelType.TIMOSHENKO_BEAM,
        ]:
            assert len(x.shape) == 1, f"Invalid shape {x.shape} for the domain."
            assert len(v) == 2, f"Invalid shape {v.shape} for load intensities."
            if isinstance(v[0], (float, int)) and isinstance(v[1], (float, int)):
                v = np.array(v, dtype=float)
                evaluator = rhs_line_const
            else:
                evaluator = rhs_line_1d_mc
        else:  # pragma: no cover
            raise NotImplementedError

        return evaluator(problem.size, problem.shape, v, x)
