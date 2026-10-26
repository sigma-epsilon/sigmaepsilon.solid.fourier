from typing import Iterable

from numpy import ndarray

from ..preproc import rhs_disk_mc, rhs0_disk_mc
from ..protocols import NavierProblemProtocol
from .loads import LoadCase
from ..config import Config

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
        return self._rhs(problem, rhs_disk_mc)

    def rhs0(self, problem: NavierProblemProtocol) -> ndarray:
        """
        Returns the zero-order coefficients of the moment loads.

        See :func:`~sigmaepsilon.solid.fourier.loads.LoadCase.rhs0` for details.
        """
        return self._rhs(problem, rhs0_disk_mc)

    def _rhs(self, problem: NavierProblemProtocol, evaluator) -> ndarray:
        assert problem.model_type.is_2d, f"Invalid model type {problem.model_type}."
        n_MC = self._num_mc or Config.NUM_MC_SAMPLES_PLATE
        return evaluator(
            problem.size, problem.shape, self.domain, self.value, n_MC=n_MC
        )
