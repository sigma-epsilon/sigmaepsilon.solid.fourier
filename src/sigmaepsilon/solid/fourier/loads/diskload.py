from typing import Iterable

import numpy as np
from numpy import ndarray

from ..preproc import rhs_disk_mc
from ..protocols import NavierProblemProtocol
from .loads import LoadCase
from ..enums import MechanicalModelType

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
            3d float array of shape (1, H, 3), where H is the total number
            of harmonic terms involved (defined for the problem). The first
            axis is always of length 1, as there is only one left hand side.
        """
        assert problem.model_type in [
            MechanicalModelType.KIRCHHOFF_LOVE_PLATE,
            MechanicalModelType.UFLYAND_MINDLIN_PLATE,
        ], f"Invalid model type {problem.model_type}."
        return rhs_disk_mc(problem.size, problem.shape, self.domain, self.value)
