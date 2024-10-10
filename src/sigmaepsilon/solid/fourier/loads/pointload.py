import numpy as np
from numpy import ndarray

from ..preproc import rhs_conc_1d, rhs_conc_2d
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d

__all__ = ["PointLoad"]


class PointLoad(LoadCase[float | Float1d, Float1d]):
    """
    A class to handle concentrated loads.

    Parameters
    ----------
    domain: float or :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        The point of application. A scalar for a beam, an iterable of
        length 2 for a plate.
    value: :class:`~sigmaepsilon.solid.fourier.loads.Float1d`
        Load values for each dof. The order of the dofs for a beam
        is [F, M], for a plate it is [F, Mx, My].

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
            the attached problem of the object is used. Default is ``None``.

        Returns
        -------
        numpy.ndarray
            2d float array of shape (H, 3), where H is the total number
            of harmonic terms involved (defined for the problem).
        """
        x = self.domain
        v = np.array(self.value)
        if hasattr(problem, "size"):
            return rhs_conc_2d(problem.size, problem.shape, v, x)
        else:
            return rhs_conc_1d(problem.length, problem.N, v, x)
