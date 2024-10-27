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
        values = np.array(self.value, dtype=float)

        if problem.model_type.is_2d:
            domain = np.array(self.domain, dtype=float)
            evaluator = rhs_conc_2d
        elif problem.model_type.is_1d:
            domain = float(self.domain)
            evaluator = rhs_conc_1d
        else:  # pragma: no cover
            raise NotImplementedError

        return evaluator(problem.size, problem.shape, domain, values)
