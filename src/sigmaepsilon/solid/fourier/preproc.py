from typing import Iterable, Union, Tuple
from numbers import Number

import numpy as np
from numba import njit, prange
from numpy import ndarray, sin, cos, ndarray, pi as PI, average as avg

from sigmaepsilon.math import atleast1d, atleast2d, atleast3d
from sigmaepsilon.math.function import Function
from sigmaepsilon.math.linalg import linspace

from .utils import sin1d, cos1d
from .config import config


def lhs_Navier(
    size: Union[float, Tuple[float]],
    shape: Union[int, Tuple[int]],
    *,
    D: Union[float, ndarray],
    S: Union[float, ndarray, None] = None,
    **kw,
) -> ndarray:
    """
    Returns coefficient matrices for a Navier solution, for a single or
    multiple left-hand sides.

    Parameters
    ----------
    size: Union[float, Tuple[float]]
        The size of the problem. Scalar for a beam, 2-tuple for a plate.
    shape: Union[int, Tuple[int]]
        The number of harmonic terms used. Scalar for a beam, 2-tuple for a plate.
    D: Union[float, ndarray]
        2d or 3d float array of bending stiffnesses for a plate, scalar or 1d float array
        for a beam.
    S: Union[float, ndarray], Optional
        2d or 3d float array of shear stiffnesses for a plate, scalar or 1d float array
        for a beam. Only for Mindlin-Reissner plates and Euler-Bernoulli beams.
        plates. Default is None.

    Note
    ----
    Shear stiffnesses must include shear correction.

    Returns
    -------
    numpy.ndarray
        The coefficients as an array. See the documentation of the corresponding
        function for further details.

    See Also
    --------
    :func:`lhs_Navier_Mindlin`
    :func:`lhs_Navier_Kirchhoff`
    :func:`lhs_Navier_Bernoulli`
    :func:`lhs_Navier_Timoshenko`
    """
    if isinstance(shape, Iterable):  # plate problem
        if S is None:
            return lhs_Navier_Kirchhoff(size, shape, atleast3d(D))
        else:
            return lhs_Navier_Mindlin(size, shape, atleast3d(D), atleast3d(S))
    else:  # beam problem
        if S is None:
            return lhs_Navier_Bernoulli(size, shape, atleast1d(D))
        else:
            return lhs_Navier_Timoshenko(size, shape, atleast1d(D), atleast1d(S))


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Mindlin(size: tuple, shape: tuple, D: ndarray, S: ndarray) -> ndarray:
    """
    JIT compiled function, that returns coefficient matrices for a Navier
    solution for multiple left-hand sides.

    Parameters
    ----------
    size: tuple
        Tuple of floats, containing the sizes of the rectagle.
    shape: tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.
    D: numpy.ndarray
        3d float array of bending stiffnesses.
    S: numpy.ndarray
        3d float array of shear stiffnesses.

    Note
    ----
    The shear stiffness values must include the shear correction factor.

    Returns
    -------
    numpy.ndarray
        4d float array of coefficients.
    """
    Lx, Ly = size
    nLHS = D.shape[0]
    M, N = shape
    res = np.zeros((nLHS, M * N, 3, 3), dtype=D.dtype)
    for iLHS in prange(nLHS):
        D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], D[iLHS, 1, 1], D[iLHS, 2, 2]
        S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                # sum Fx
                res[iLHS, iMN, 0, 0] = (
                    PI**2 * S44 * n**2 / Ly**2 + PI**2 * S55 * m**2 / Lx**2
                )
                res[iLHS, iMN, 0, 1] = -PI * S44 * n / Ly
                res[iLHS, iMN, 0, 2] = PI * S55 * m / Lx
                # sum Mx
                res[iLHS, iMN, 1, 0] = res[iLHS, iMN, 0, 1]
                res[iLHS, iMN, 1, 1] = (
                    (PI**2) * D22 * n**2 / Ly**2 + PI**2 * D66 * m**2 / Lx**2 + S44
                )
                res[iLHS, iMN, 1, 2] = -(PI**2) * D12 * m * n / (
                    Lx * Ly
                ) - PI**2 * D66 * m * n / (Lx * Ly)
                # sum My
                res[iLHS, iMN, 2, 0] = res[iLHS, iMN, 0, 2]
                res[iLHS, iMN, 2, 1] = res[iLHS, iMN, 1, 2]
                res[iLHS, iMN, 2, 2] = (
                    PI**2 * D11 * m**2 / Lx**2 + PI**2 * D66 * n**2 / Ly**2 + S55
                )
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Kirchhoff(size: tuple, shape: tuple, D: ndarray) -> ndarray:
    """
    JIT compiled function, that returns coefficient matrices for a Navier
    solution for multiple left-hand sides.

    Parameters
    ----------
    size: tuple
        Tuple of floats, containing the sizes of the rectagle.
    shape: tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.
    D: numpy.ndarray
        3d float array of bending stiffnesses.

    Returns
    -------
    numpy.ndarray
        2d float array of coefficients.
    """
    Lx, Ly = size
    nLHS = D.shape[0]
    M, N = shape
    res = np.zeros((nLHS, M * N), dtype=D.dtype)
    for iLHS in prange(nLHS):
        D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], D[iLHS, 1, 1], D[iLHS, 2, 2]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iLHS, iMN] = (
                    PI**4 * D11 * m**4 / Lx**4
                    + 2 * PI**4 * D12 * m**2 * n**2 / (Lx**2 * Ly**2)
                    + PI**4 * D22 * n**4 / Ly**4
                    + 4 * PI**4 * D66 * m**2 * n**2 / (Lx**2 * Ly**2)
                )
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Bernoulli(L: float, N: int, EI: ndarray) -> ndarray:
    """
    JIT compiled function, that returns coefficient matrices for a Navier
    solution for multiple left-hand sides.

    Parameters
    ----------
    L: float
        The length of the beam.
    N: int
        The number of harmonic terms.
    EI: numpy.ndarray
        1d float array of bending stiffnesses.

    Returns
    -------
    numpy.ndarray
        2d float array of coefficients.
    """
    nLHS = EI.shape[0]
    res = np.zeros((nLHS, N), dtype=EI.dtype)
    for iLHS in prange(nLHS):
        for n in prange(1, N + 1):
            res[iLHS, n - 1] = PI**4 * EI[iLHS] * n**4 / L**4
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Timoshenko(L: float, N: int, EI: ndarray, GA: ndarray) -> ndarray:
    """
    JIT compiled function, that returns coefficient matrices for a Navier
    solution for multiple left-hand sides.

    Parameters
    ----------
    L: float
        The length of the beam.
    N: int
        The number of harmonic terms.
    EI: numpy.ndarray
        1d float array of bending stiffnesses.
    GA: numpy.ndarray
        1d float array of shear stiffnesses.

    Note
    ----
    The shear stiffness values must include the shear correction factor.

    Returns
    -------
    numpy.ndarray
        4d float array of coefficients.
    """
    nLHS = EI.shape[0]
    res = np.zeros((nLHS, N, 2, 2), dtype=EI.dtype)
    for iLHS in prange(nLHS):
        for n in prange(1, N + 1):
            iN = n - 1
            c1 = PI * n / L
            c2 = c1 * PI * n / L
            res[iLHS, iN, 0, 0] = c2 * GA[iLHS]
            res[iLHS, iN, 0, 1] = -c1 * GA[iLHS]
            res[iLHS, iN, 1, 0] = res[iLHS, iN, 0, 1]
            res[iLHS, iN, 1, 1] = GA[iLHS] + c2 * EI[iLHS]
    return res


# RIGHT HAND SIDES


@njit(nogil=True, parallel=True, cache=True)
def rhs_Kirchhoff(coeffs: ndarray, size: tuple) -> ndarray:
    """
    Calculates unknowns for Kirchhoff plates.
    """
    Lx, Ly = size
    nRHS, N = coeffs.shape[:2]
    res = np.zeros((nRHS, N))
    cx = PI / Lx
    cy = PI / Ly
    for i in prange(nRHS):
        for n in prange(N):
            res[i, n] = (
                coeffs[i, n, 0]
                - coeffs[i, n, 1] * cy * (n + 1)
                - coeffs[i, n, 2] * cx * (n + 1)
            )
    return res


@njit(nogil=True, parallel=True, cache=True)
def rhs_Bernoulli(coeffs: ndarray, L: float) -> ndarray:
    """
    Calculates unknowns for Bernoulli beams.
    """
    nRHS, N = coeffs.shape[:2]
    res = np.zeros((nRHS, N))
    c = PI / L
    for i in prange(nRHS):
        for n in prange(N):
            res[i, n] = coeffs[i, n, 0] - coeffs[i, n, 1] * c * (n + 1)
    return res


def rhs_line_const(L: float, N: int, v: ndarray, x: ndarray) -> ndarray:
    """
    Returns coefficients for constant loads over line segments.
    Values are expected in the order [f, m].
    """
    return _line_const_(L, N, atleast2d(x), atleast2d(v))


def rhs_line_1d_any(L: float, N: int, v: Iterable, x: ndarray) -> ndarray:
    """
    Returns coefficients for arbitrary loads over line segments.
    Values are expected in the order [f, m].
    """
    rhs = np.zeros((1, N, 2), dtype=x.dtype)

    if isinstance(v[0], Number) and isinstance(v[1], Number):
        v = np.array(v, dtype=float)
        return rhs_line_const(L, N, v, x)

    num_MC_samples = config.get("num_MC_samples", 1000)

    points = np.linspace(x[0], x[1], num_MC_samples)
    d = x[1] - x[0]
    multi = (2 / L) * d

    for i in range(2):
        trigfnc = sin1d if i == 0 else cos1d
        value = v[i]
        if isinstance(value, str):
            f = Function(value, variables=["x"], dim=1)
            rhs[0, :, i] = list(
                map(
                    lambda i: multi * avg(trigfnc(points, i, L) * f([points])),
                    np.arange(1, N + 1),
                )
            )
        elif isinstance(value, Number):
            _v = np.array([0, 0], dtype=float)
            _v[i] = value
            rhs[0, :, i] = rhs_line_const(L, N, _v, x)[0, :, i]

    return rhs


@njit(nogil=True, parallel=True, cache=True)
def _line_const_(L: float, N: int, x: ndarray, values: ndarray) -> ndarray:
    nR = values.shape[0]
    rhs = np.zeros((nR, N, 2), dtype=x.dtype)
    for iR in prange(nR):
        for n in prange(1, N + 1):
            iN = n - 1
            xa, xb = x[iR]
            f, m = values[iR]
            c = PI * n / L
            rhs[iR, iN, 0] = (2 * f / (PI * n)) * (cos(c * xa) - cos(c * xb))
            rhs[iR, iN, 1] = (2 * m / (PI * n)) * (sin(c * xb) - sin(c * xa))
    return rhs


def rhs_line_2d_any(size: tuple, shape: tuple, v: Iterable, x: Iterable) -> ndarray:
    """
    Returns coefficients for arbitrary line loads for 2d problems.
    Values are expected in the order [f, mx, my].
    """
    M, N = shape
    Lx, Ly = size
    rhs = np.zeros((1, M * N, 3), dtype=float)

    num_MC_samples = config.get("num_MC_samples", 1000)

    points = linspace(x[0], x[1], num_MC_samples)
    d = np.linalg.norm(x[1] - x[0])
    multi = (4 / Lx / Ly) * d
    px = points[:, 0]
    py = points[:, 1]

    for i in range(3):
        if i == 0:
            trigfnc = lambda x, y, i, j, Lx, Ly: sin1d(x, i, Lx) * sin1d(y, j, Ly)
        elif i == 1:
            trigfnc = lambda x, y, i, j, Lx, Ly: sin1d(x, i, Lx) * cos1d(y, j, Ly)
        elif i == 2:
            trigfnc = lambda x, y, i, j, Lx, Ly: cos1d(x, i, Lx) * sin1d(y, j, Ly)

        if isinstance(v[i], str):
            f = Function(v[i], variables=["x y"], dim=2)
            f_vals = f(points)
        elif isinstance(v[i], Number):
            f_vals = v[i]

        for n in range(1, N + 1):
            for m in range(1, M + 1):
                mn = (m - 1) * N + n - 1
                rhs[0, mn, i] = multi * avg(trigfnc(px, py, n, m, Lx, Ly) * f_vals)

    return rhs


def rhs_rect_const(size: tuple, shape: tuple, x: ndarray, v: ndarray) -> ndarray:
    """
    Returns coefficients for constant loads over rectangular patches
    in the order [f, mx, my].
    """
    return _rect_const_(size, shape, atleast2d(v), atleast3d(x))


@njit(nogil=True, cache=True)
def __rect_const__(
    size: tuple,
    m: int,
    n: int,
    xc: float,
    yc: float,
    w: float,
    h: float,
    values: ndarray,
) -> ndarray:
    Lx, Ly = size
    f, mx, my = values
    return np.array(
        [
            16
            * f
            * sin(PI * m * w / (Lx * 2))
            * sin(PI * m * xc / Lx)
            * sin(PI * h * n / (Ly * 2))
            * sin(PI * n * yc / Ly)
            / (PI**2 * m * n),
            16
            * mx
            * sin(PI * m * w / (Lx * 2))
            * sin(PI * m * xc / Lx)
            * sin(PI * h * n / (Ly * 2))
            * cos(PI * n * yc / Ly)
            / (PI**2 * m * n),
            16
            * my
            * sin(PI * m * w / (Lx * 2))
            * cos(PI * m * xc / Lx)
            * sin(PI * h * n / (Ly * 2))
            * sin(PI * n * yc / Ly)
            / (PI**2 * m * n),
        ]
    )


@njit(nogil=True, parallel=True, cache=True)
def _rect_const_(
    size: tuple, shape: tuple, values: ndarray, points: ndarray
) -> ndarray:
    nRect = values.shape[0]  # number of rectangles
    M, N = shape
    rhs = np.zeros((nRect, M * N, 3), dtype=points.dtype)
    for iRect in prange(nRect):
        xmin, ymin = points[iRect, 0]
        xmax, ymax = points[iRect, 1]
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = np.abs(xmax - xmin)
        h = np.abs(ymax - ymin)
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iRect, mn, :] = __rect_const__(
                    size, m, n, xc, yc, w, h, values[iRect]
                )
    return rhs


def rhs_conc_1d(L: float, N: int, v: ndarray, x: ndarray) -> ndarray:
    return _conc1d_(L, N, atleast2d(v), atleast1d(x))


@njit(nogil=True, parallel=True, cache=True)
def _conc1d_(L: tuple, N: tuple, values: ndarray, points: ndarray) -> ndarray:
    nRHS = values.shape[0]  # number of point loads
    c = 2 / L
    rhs = np.zeros((nRHS, N, 2), dtype=points.dtype)
    PI = np.pi
    for iRHS in prange(nRHS):
        x = points[iRHS]
        f, m = values[iRHS]
        Sx = PI * x / L
        for n in prange(1, N + 1):
            i = n - 1
            rhs[iRHS, i, 0] = c * f * sin(n * Sx)
            rhs[iRHS, i, 1] = c * m * cos(n * Sx)
    return rhs


def rhs_conc_2d(size: tuple, shape: tuple, v: ndarray, x: ndarray) -> ndarray:
    return _conc2d_(size, shape, atleast2d(v), atleast2d(x))


@njit(nogil=True, parallel=True, cache=True)
def _conc2d_(size: tuple, shape: tuple, values: ndarray, points: ndarray) -> ndarray:
    nRHS = values.shape[0]  # number of point loads
    Lx, Ly = size
    c = 4 / Lx / Ly
    M, N = shape
    rhs = np.zeros((nRHS, M * N, 3), dtype=points.dtype)
    PI = np.pi
    for iRHS in prange(nRHS):
        x, y = points[iRHS]
        fz, mx, my = values[iRHS]
        Sx = PI * x / Lx
        Sy = PI * y / Ly
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iRHS, mn, 0] = c * fz * sin(m * Sx) * sin(n * Sy)
                rhs[iRHS, mn, 1] = c * mx * cos(m * Sx) * sin(n * Sy)
                rhs[iRHS, mn, 2] = c * my * sin(m * Sx) * cos(n * Sy)
    return rhs
