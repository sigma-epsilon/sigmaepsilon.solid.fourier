from typing import Iterable, Union, Tuple
from types import NoneType
from functools import partial

import numpy as np
from numba import njit, prange
from numpy import ndarray, sin, cos

from sigmaepsilon.math import atleast1d, atleast3d

from .mc import _monte_carlo_1d, _monte_carlo_2d
from .utils import (
    generate_random_points_in_disk,
    generate_random_points_in_rectangle,
    generate_random_points_on_line_segment_1d,
    generate_random_points_on_line_segment_2d,
)


def lhs_Navier(
    size: Union[float, Tuple[float]],
    shape: Union[int, Tuple[int]],
    *,
    D: Union[float, ndarray],
    S: Union[float, ndarray, NoneType] = None,
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
    PI = np.pi
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
    PI = np.pi
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
    PI = np.pi
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
    PI = np.pi
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
    PI = np.pi
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
    c = np.pi / L
    for i in prange(nRHS):
        for n in prange(N):
            res[i, n] = coeffs[i, n, 0] - coeffs[i, n, 1] * c * (n + 1)
    return res


@njit(nogil=True, parallel=True, cache=True)
def _rhs_line_const(L: float, N: int, domain: ndarray, values: ndarray) -> ndarray:
    PI = np.pi
    rhs = np.zeros((N, 2), dtype=domain.dtype)
    for n in prange(1, N + 1):
        iN = n - 1
        xa, xb = domain
        f, m = values
        c = PI * n / L
        rhs[iN, 0] = (2 * f / (PI * n)) * (cos(c * xa) - cos(c * xb))
        rhs[iN, 1] = (2 * m / (PI * n)) * (sin(c * xb) - sin(c * xa))
    return rhs


@njit(nogil=True, cache=True)
def _rhs_rect_const_single_harmonic(
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
    PI = np.pi
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
def _rhs_rect_const(
    size: tuple, shape: tuple, domain: ndarray, values: ndarray
) -> ndarray:
    M, N = shape
    rhs = np.zeros((M * N, 3), dtype=domain.dtype)
    xmin, ymin = domain[0]
    xmax, ymax = domain[1]
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = np.abs(xmax - xmin)
    h = np.abs(ymax - ymin)
    for m in prange(1, M + 1):
        for n in prange(1, N + 1):
            mn = (m - 1) * N + n - 1
            rhs[mn, :] = _rhs_rect_const_single_harmonic(
                size, m, n, xc, yc, w, h, values
            )
    return rhs


@njit(nogil=True, parallel=True, cache=True)
def _rhs_conc_1d(L: tuple, N: tuple, point: float, values: ndarray) -> ndarray:
    c = 2 / L
    rhs = np.zeros((N, 2), dtype=values.dtype)
    PI = np.pi
    f, m = values
    Sx = PI * point / L
    for n in prange(1, N + 1):
        iN = n - 1
        rhs[iN, 0] = c * f * sin(n * Sx)
        rhs[iN, 1] = c * m * cos(n * Sx)
    return rhs


@njit(nogil=True, parallel=True, cache=True)
def _rhs_conc_2d(size: tuple, shape: tuple, point: ndarray, values: ndarray) -> ndarray:
    Lx, Ly = size
    M, N = shape
    rhs = np.zeros((M * N, 3), dtype=point.dtype)
    x, y = point
    fz, mx, my = values
    Sx = np.pi * x / Lx
    Sy = np.pi * y / Ly
    c = 4 / Lx / Ly
    for m in prange(1, M + 1):
        for n in prange(1, N + 1):
            mn = (m - 1) * N + n - 1
            rhs[mn, 0] = c * fz * sin(m * Sx) * sin(n * Sy)
            rhs[mn, 1] = c * mx * cos(m * Sx) * sin(n * Sy)
            rhs[mn, 2] = c * my * sin(m * Sx) * cos(n * Sy)
    return rhs


def rhs_conc_1d(L: float, N: int, domain: float, values: ndarray) -> ndarray:
    """
    Returns coefficients for a concentrated load on a beam.
    Load values are expected in the order [f, m].
    """
    return _rhs_conc_1d(L, N, domain, values)


def rhs_conc_2d(size: tuple, shape: tuple, domain: ndarray, values: ndarray) -> ndarray:
    """
    Returns coefficients for a concentrated load on a plate.
    Load values are expected in the order [f, mx, my].
    """
    return _rhs_conc_2d(size, shape, domain, values)


def rhs_line_const(L: float, N: int, domain: ndarray, values: ndarray) -> ndarray:
    """
    Returns coefficients for constant loads over line segments.
    Values are expected in the order [f, m].
    """
    return _rhs_line_const(L, N, domain, values)


def rhs_line_1d_mc(
    size: float,
    shape: int,
    domain: ndarray,
    values: Iterable,
    *,
    n_MC: int | NoneType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over line segments.
    Values are expected in the order [f, m].
    """
    length = domain[1] - domain[0]
    rhs = np.zeros((shape, 2), dtype=float)
    rpg = partial(generate_random_points_on_line_segment_1d, domain[0], domain[1])
    _monte_carlo_1d(size, shape, values, length, rpg=rpg, n_MC=n_MC, out=rhs)
    return rhs


def rhs_line_2d_mc(
    size: tuple,
    shape: tuple,
    domain: ndarray,
    values: Iterable,
    *,
    n_MC: int | NoneType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary line loads for 2d problems.
    Values are expected in the order [f, mx, my].
    """
    length = np.linalg.norm(domain[1] - domain[0])
    rhs = np.zeros((np.prod(shape), 3), dtype=float)
    rpg = partial(generate_random_points_on_line_segment_2d, domain[0], domain[1])
    _monte_carlo_2d(size, shape, values, length, rpg=rpg, n_MC=n_MC, out=rhs)
    return rhs


def rhs_rect_const(
    size: tuple, shape: tuple, domain: ndarray, values: ndarray
) -> ndarray:
    """
    Returns coefficients for constant loads over rectangular patches
    in the order [f, mx, my].
    """
    return _rhs_rect_const(size, shape, domain, values)


def rhs_rect_mc(
    size: tuple,
    shape: tuple,
    domain: ndarray,
    values: Iterable,
    *,
    n_MC: int | NoneType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over a rectangle.
    Load values are expected in the order [f, mx, my].
    """
    area = np.abs(domain[1, 0] - domain[0, 0]) * np.abs(domain[1, 1] - domain[0, 1])
    rhs = np.zeros((np.prod(shape), 3), dtype=float)
    rpg = partial(generate_random_points_in_rectangle, domain[0], domain[1])
    _monte_carlo_2d(size, shape, values, area, rpg=rpg, n_MC=n_MC, out=rhs)
    return rhs


def rhs_disk_mc(
    size: tuple,
    shape: tuple,
    domain: ndarray,
    values: Iterable,
    *,
    n_MC: int | NoneType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over a disk.
    Load values are expected in the order [f, mx, my].
    """
    center = domain[0]
    radius = domain[1]
    area = np.pi * radius**2
    rhs = np.zeros((np.prod(shape), 3), dtype=float)
    rpg = partial(generate_random_points_in_disk, center, radius)
    _monte_carlo_2d(size, shape, values, area, rpg=rpg, n_MC=n_MC, out=rhs)
    return rhs
