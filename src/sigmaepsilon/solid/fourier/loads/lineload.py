from numbers import Number

import numpy as np
from numpy import ndarray, average as avg

from sigmaepsilon.math.function import Function

from ..preproc import rhs_line_const
from ..utils import sin1d, cos1d
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
        x = np.array(self.domain, dtype=float)
        v = self.value
        if len(x.shape) == 1:
            assert len(v) == 2, f"Invalid shape {v.shape} for load intensities."
            if isinstance(v[0], Number) and isinstance(v[1], Number):
                v = np.array(v, dtype=float)
                return rhs_line_const(problem.length, problem.N, v, x)
            else:
                rhs = np.zeros((1, problem.N, 2), dtype=x.dtype)
                if isinstance(v[0], str):
                    f = Function(v[0], variables=["x"], dim=1)
                    L = problem.length
                    points = np.linspace(x[0], x[1], 1000)
                    d = x[1] - x[0]
                    rhs[0, :, 0] = list(
                        map(
                            lambda i: (2 / L)
                            * d
                            * avg(sin1d(points, i, L) * f([points])),
                            np.arange(1, problem.N + 1),
                        )
                    )
                elif isinstance(v[0], Number):
                    _v = np.array([v[0], 0], dtype=float)
                    rhs[0, :, 0] = rhs_line_const(problem.length, problem.N, _v, x)[
                        0, :, 0
                    ]
                if isinstance(v[1], str):
                    f = Function(v[1], variables=["x"], dim=1)
                    L = problem.length
                    points = np.linspace(x[0], x[1], 1000)
                    d = x[1] - x[0]
                    rhs[0, :, 1] = list(
                        map(
                            lambda i: (2 / L)
                            * d
                            * avg(cos1d(points, i, L) * f([points])),
                            np.arange(1, problem.N + 1),
                        )
                    )
                elif isinstance(v[1], Number):
                    _v = np.array([0, v[1]], dtype=float)
                    rhs[0, :, 1] = rhs_line_const(problem.length, problem.N, _v, x)[
                        0, :, 1
                    ]
                return rhs
        else:
            raise NotImplementedError(
                "Line loads are only implemented for 1d problems at the moment."
            )
