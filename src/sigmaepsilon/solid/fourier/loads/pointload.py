import numpy as np
from numpy import ndarray

from ..preproc import rhs_conc_1d, rhs_conc_2d
from ..protocols import NavierProblemProtocol
from .loads import LoadCase, Float1d
from ..enums import MechanicalModelType

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
            3d float array of shape (1, H, 3), where H is the total number
            of harmonic terms involved (defined for the problem). The first
            axis is always of length 1, as there is only one left hand side.
        """
        x = np.array(self.domain)
        v = np.array(self.value)

        if problem.model_type in [
            MechanicalModelType.KIRCHHOFF_LOVE_PLATE,
            MechanicalModelType.UFLYAND_MINDLIN_PLATE,
        ]:
            evaluator = rhs_conc_2d
        elif problem.model_type in [
            MechanicalModelType.BERNOULLI_EULER_BEAM,
            MechanicalModelType.TIMOSHENKO_BEAM,
        ]:
            evaluator = rhs_conc_1d
        else:  # pragma: no cover
            raise NotImplementedError
        
        x = np.atleast_1d(x)
        return evaluator(problem.size, problem.shape, v, x)
