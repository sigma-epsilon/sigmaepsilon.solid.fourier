from typing import Iterable
from types import NoneType

import numpy as np
from numpy import sin, cos, ndarray, pi as PI
from numba import njit, prange

from sigmaepsilon.math import atleast1d, atleast2d, atleast3d, atleast4d, itype_of_ftype


__all__ = ["postproc", "postproc_Mindlin", "postproc_Kirchhoff"]


UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(range(13))


def postproc(
    size: float | Iterable[float],
    shape: int | Iterable[int],
    points: ndarray,
    solution: ndarray,
    loads: ndarray,
    D: float | ndarray,
    S: float | ndarray | NoneType = None,
) -> ndarray:
    """
    Calculates postprocessing items.
    """
    ftype = solution.dtype
    itype = itype_of_ftype(ftype)
    if S is not None:
        if not isinstance(size, Iterable):
            res = postproc_Timoshenko(
                size,
                shape,
                atleast1d(points).astype(ftype),
                atleast3d(solution).astype(ftype),
                atleast1d(D),
                atleast1d(S),
            )
        else:
            res = postproc_Mindlin(
                np.array(size).astype(ftype),
                np.array(shape).astype(itype),
                atleast2d(points).astype(ftype),
                atleast4d(solution).astype(ftype),
                atleast3d(D),
                atleast3d(S),
            )
    else:
        if not isinstance(size, Iterable):
            res = postproc_Bernoulli(
                size,
                shape,
                atleast1d(points).astype(ftype),
                atleast3d(solution).astype(ftype),
                atleast1d(D),
                loads,
            )
        else:
            res = postproc_Kirchhoff(
                np.array(size).astype(ftype),
                np.array(shape).astype(itype),
                atleast2d(points).astype(ftype),
                atleast3d(solution).astype(ftype),
                atleast3d(D),
                loads,
            )
    # (N, nLHS, nRHS, nP, ...)
    res = np.sum(res, axis=0)
    # (nLHS, nRHS, nP, ...)
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Bernoulli(
    L: float, N: int, points: ndarray, solution: ndarray, EI: ndarray, loads
) -> ndarray:
    """
    JIT-compiled function that calculates post-processing quantities
    at selected ponts for multiple left- and right-hand sides.

    Parameters
    ----------
    L: float
        The length of the beam.
    N: int
        The number of harmonic terms.
    points: numpy.ndarray
        1d array of shape (nP,) of coordinates.
    solution: numpy.ndarray
        Results of a Navier solution as a 3d array of shape (nRHS, nLHS, N).
    EI: float
        Bending stiffnesses as a 3d float array of shape (nLHS, 3, 3).

    Returns
    -------
    numpy.ndarray
        5d array of shape (N, nLHS, nRHS, nP, ...) of post-processing items.
        The indices along the last axis denote the following quantities:

            0 : displacement

            1 : rotation

            2 : curvature

            3 : shear strain

            4 : moment

            5 : shear force

    """
    nP = points.shape[0]
    nLHS, nRHS = solution.shape[:2]
    res = np.zeros((N, nLHS, nRHS, nP, 6), dtype=solution.dtype)
    for iRHS in prange(nRHS):
        for iLHS in prange(nLHS):
            for iP in prange(nP):
                x = points[iP]
                for n in prange(1, N + 1):
                    iN = n - 1
                    arg = PI * n / L
                    Sn = sin(x * arg)
                    Cn = cos(x * arg)
                    vn = solution[iLHS, iRHS, iN]
                    q = loads[iRHS, iN, 1]
                    res[iN, iLHS, iRHS, iP, 0] = vn * Sn
                    res[iN, iLHS, iRHS, iP, 1] = vn * arg * Cn
                    res[iN, iLHS, iRHS, iP, 2] = -vn * arg**2 * Sn
                    res[iN, iLHS, iRHS, iP, 4] = -EI[iLHS] * vn * arg**2 * Sn
                    res[iN, iLHS, iRHS, iP, 5] = (EI[iLHS] * vn * arg**3 + q) * Cn
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Timoshenko(
    L: float, N: int, points: ndarray, solution: ndarray, EI: ndarray, GA: ndarray
) -> ndarray:
    """
    JIT-compiled function that calculates post-processing quantities
    at selected ponts for multiple left- and right-hand sides.

    Parameters
    ----------
    L: float
        The length of the beam.
    N: int
        The number of harmonic terms.
    points: numpy.ndarray
        1d array of shape (nP,) of coordinates.
    solution: numpy.ndarray
        Results of a Navier solution as a 4d array of shape (nRHS, nLHS, N, 2).
    EI: float
        Bending stiffnesses as an 1d float array of shape (nLHS).
    GA: float
        Corrected shear stiffnesses as an 1d float array of shape (nLHS).

    Returns
    -------
    numpy.ndarray
        5d array of shape (N, nLHS, nRHS, nP, ...) of post-processing items.
        The indices along the last axis denote the following quantities:

            0 : displacement

            1 : rotation

            2 : curvature

            3 : shear strain

            4 : moment

            5 : shear force

    """
    nP = points.shape[0]
    nLHS, nRHS = solution.shape[:2]
    res = np.zeros((N, nLHS, nRHS, nP, 6), dtype=solution.dtype)
    for iRHS in prange(nRHS):
        for iLHS in prange(nLHS):
            for iP in prange(nP):
                x = points[iP]
                for n in prange(1, N + 1):
                    iN = n - 1
                    arg = PI * n / L
                    Sn = sin(x * arg)
                    Cn = cos(x * arg)
                    vn, rn = solution[iLHS, iRHS, iN]
                    res[iN, iLHS, iRHS, iP, 0] = vn * Sn
                    res[iN, iLHS, iRHS, iP, 1] = rn * Cn
                    res[iN, iLHS, iRHS, iP, 2] = -rn * arg * Sn
                    res[iN, iLHS, iRHS, iP, 3] = (vn * arg - rn) * Cn
                    res[iN, iLHS, iRHS, iP, 4] = -EI[iLHS] * rn * arg * Sn
                    res[iN, iLHS, iRHS, iP, 5] = GA[iLHS] * (vn * arg - rn) * Cn
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Mindlin(
    size, shape: ndarray, points: ndarray, solution: ndarray, D: ndarray, S: ndarray
) -> ndarray:
    """
    JIT-compiled function that calculates post-processing quantities
    at selected ponts for multiple left- and right-hand sides.

    Parameters
    ----------
    size: numpy.ndarray
        Sizes in both directions as an 1d float array of length 2.
    shape: numpy.ndarray
        Number of harmonic terms involved in both directions as an
        1d integer array of length 2.
    points: numpy.ndarray
        2d array of shape (nP, 2) of coordinates.
    solution: numpy.ndarray
        Results of a Navier solution as a 4d array of shape (nRHS, nLHS, M * N, 3).
    D: numpy.ndarray
        Bending stiffnesses as a 3d float array of shape (nLHS, 3, 3).
    S: numpy.ndarray
        Corrected shear stiffness as a 3d float array of shape (nLHS, 2, 2).

    Returns
    -------
    numpy.ndarray
        5d array of shape (M * N, nRHS, nLHS, nP, ...) of post-processing items.
        The indices along the last axis denote the following quantities:

            0 : displacement z

            1 : rotation x

            2 : rotation y

            3 : curvature x

            4 : curvature y

            5 : curvature xy

            6 : shear strain xz

            7 : shear strain yz

            8 : moment x

            9 : moment y

            10 : moment xy

            11 : shear force x

            12 : shear force y

    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nLHS, nRHS = solution.shape[:2]
    res = np.zeros((M * N, nLHS, nRHS, nP, 13), dtype=D.dtype)
    for iRHS in prange(nRHS):
        for iLHS in prange(nLHS):
            D11, D12, D22, D66 = (
                D[iLHS, 0, 0],
                D[iLHS, 0, 1],
                D[iLHS, 1, 1],
                D[iLHS, 2, 2],
            )
            S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
            for iP in prange(nP):
                xp, yp = points[iP, :2]
                for m in prange(1, M + 1):
                    Sm = sin(PI * m * xp / Lx)
                    Cm = cos(PI * m * xp / Lx)
                    for n in prange(1, N + 1):
                        iMN = (m - 1) * N + n - 1
                        Amn, Bmn, Cmn = solution[iLHS, iRHS, iMN]
                        Sn = sin(PI * n * yp / Ly)
                        Cn = cos(PI * n * yp / Ly)
                        res[iMN, iLHS, iRHS, iP, UZ] = Amn * Sm * Sn
                        res[iMN, iLHS, iRHS, iP, ROTX] = Bmn * Sm * Cn
                        res[iMN, iLHS, iRHS, iP, ROTY] = Cmn * Sn * Cm
                        res[iMN, iLHS, iRHS, iP, CX] = -PI * Cmn * m * Sm * Sn / Lx
                        res[iMN, iLHS, iRHS, iP, CY] = PI * Bmn * n * Sm * Sn / Ly
                        res[iMN, iLHS, iRHS, iP, CXY] = (
                            -PI * Bmn * m * Cm * Cn / Lx + PI * Cmn * n * Cm * Cn / Ly
                        )
                        res[iMN, iLHS, iRHS, iP, EXZ] = (
                            Cmn * Sn * Cm + PI * Amn * m * Sn * Cm / Lx
                        )
                        res[iMN, iLHS, iRHS, iP, EYZ] = (
                            -Bmn * Sm * Cn + PI * Amn * n * Sm * Cn / Ly
                        )
                        res[iMN, iLHS, iRHS, iP, MX] = (
                            PI * Bmn * D12 * n * Sm * Sn / Ly
                            - PI * Cmn * D11 * m * Sm * Sn / Lx
                        )
                        res[iMN, iLHS, iRHS, iP, MY] = (
                            PI * Bmn * D22 * n * Sm * Sn / Ly
                            - PI * Cmn * D12 * m * Sm * Sn / Lx
                        )
                        res[iMN, iLHS, iRHS, iP, MXY] = (
                            -PI * Bmn * D66 * m * Cm * Cn / Lx
                            + PI * Cmn * D66 * n * Cm * Cn / Ly
                        )
                        res[iMN, iLHS, iRHS, iP, QX] = (
                            Cmn * S55 * Sn * Cm + PI * Amn * S55 * m * Sn * Cm / Lx
                        )
                        res[iMN, iLHS, iRHS, iP, QY] = (
                            -Bmn * S44 * Sm * Cn + PI * Amn * S44 * n * Sm * Cn / Ly
                        )
    return res


# fmt: off
@njit(nogil=True, parallel=True, cache=True)
def postproc_Kirchhoff(
    size, shape: ndarray, points: ndarray, solution: ndarray, D: ndarray, loads: ndarray
) -> ndarray:
    """
    JIT-compiled function that calculates post-processing quantities
    at selected ponts for multiple left- and right-hand sides.

    Parameters
    ----------
    size: numpy.ndarray
        Sizes in both directions as an 1d float array of length 2.
    shape: numpy.ndarray
        Number of harmonic terms involved in both directions as an
        1d integer array of length 2.
    points: numpy.ndarray
        2d array of point coordinates of shape (nP, 2).
    solution: numpy.ndarray
        results of a Navier solution as a 3d array of shape (nRHS, nLHS, M * N).
    D: numpy.ndarray
        3d array of bending stiffness terms (nLHS, 3, 3).

    Returns
    -------
    numpy.ndarray[M * N, nRHS, nLHS, nP, ...]
        numpy array of post-processing items. The indices along
        the last axpis denote the following quantities:

            0 : displacement z

            1 : rotation x

            2 : rotation y

            3 : curvature x

            4 : curvature y

            5 : curvature xy

            6 : shear strain xz

            7 : shear strain yz

            8 : moment y

            9 : moment y

            10 : moment xy

            11 : shear force x

            12 : shear force y

    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nLHS, nRHS = solution.shape[:2]
    res = np.zeros((M * N, nLHS, nRHS, nP, 13), dtype=D.dtype)
    for iRHS in prange(nRHS):
        for iLHS in prange(nLHS):
            D11, D12, D22, D66 = (
                D[iLHS, 0, 0],
                D[iLHS, 0, 1],
                D[iLHS, 1, 1],
                D[iLHS, 2, 2],
            )
            for iP in prange(nP):
                x, y = points[iP, :2]
                for m in prange(1, M + 1):
                    Sm = sin(PI * m * x / Lx)
                    Cm = cos(PI * m * x / Lx)
                    for n in prange(1, N + 1):
                        Sn = sin(PI * n * y / Ly)
                        Cn = cos(PI * n * y / Ly)
                        iMN = (m - 1) * N + n - 1
                        Amn = solution[iLHS, iRHS, iMN]
                        qmn, rmn = loads[iRHS, iMN, 1:]
                        res[iMN, iLHS, iRHS, iP, UZ] = Amn * Sm * Sn
                        res[iMN, iLHS, iRHS, iP, ROTX] = PI * Amn * n * Sm * Cn / Ly
                        res[iMN, iLHS, iRHS, iP, ROTY] = -PI * Amn * m * Sn * Cm / Lx
                        res[iMN, iLHS, iRHS, iP, CX] = (
                            PI**2 * Amn * m**2 * Sm * Sn / Lx**2
                        )
                        res[iMN, iLHS, iRHS, iP, CY] = (
                            PI**2 * Amn * n**2 * Sm * Sn / Ly**2
                        )
                        res[iMN, iLHS, iRHS, iP, CXY] = (
                            -2 * PI**2 * Amn * m * n * Cm * Cn / (Lx * Ly)
                        )
                        res[iMN, iLHS, iRHS, iP, MX] = (
                            PI**2 * Amn * D11 * m**2 * Sm * Sn / Lx**2
                            + PI**2 * Amn * D12 * n**2 * Sm * Sn / Ly**2
                        )
                        res[iMN, iLHS, iRHS, iP, MY] = (
                            PI**2 * Amn * D12 * m**2 * Sm * Sn / Lx**2
                            + PI**2 * Amn * D22 * n**2 * Sm * Sn / Ly**2
                        )
                        res[iMN, iLHS, iRHS, iP, MXY] = (
                            -2 * PI**2 * Amn * D66 * m * n * Cm * Cn / (Lx * Ly)
                        )
                        res[iMN, iLHS, iRHS, iP, QX] = (
                            PI**3 * Amn * D11 * m**3 * Sn * Cm / Lx**3
                            + PI**3 * Amn * D12 * m * n**2 * Sn * Cm / (Lx * Ly**2)
                            + 2 * PI**3 * Amn * D66 * m * n**2 * Sn * Cm / (Lx * Ly**2)
                            + rmn * Sn * Cm
                        )
                        res[iMN, iLHS, iRHS, iP, QY] = (
                            PI**3 * Amn * D12 * m**2 * n * Sm * Cn / (Lx**2 * Ly)
                            + PI**3 * Amn * D22 * n**3 * Sm * Cn / Ly**3
                            + 2 * PI**3 * Amn * D66 * m**2 * n * Sm * Cn / (Lx**2 * Ly)
                            - qmn * Sm * Cn
                        )
    return res
# fmt: on

# fmt: off
@njit(nogil=True, parallel=True, cache=True)
def eval_loads_1d(
    size: float,
    shape: int, 
    coeffs:ndarray, 
    points:ndarray,
    ) -> ndarray:
    """
    Evaluates the loads at some points for beam problems.
    """
    Lx = size
    N = shape
    nP = points.shape[0]
    nDOF = 2
    res = np.zeros((nP, nDOF), dtype=float)
    for iP in prange(nP):
        xP = points[iP]
        for n in range(1, N + 1):
            iN = n-1
            res[iP, 0] += (
                coeffs[iN, 0]
                * np.sin(xP * np.pi * n / Lx) 
            )
            res[iP, 1] += (
                coeffs[iN, 1]
                * np.cos(xP * np.pi * n / Lx) 
            )
    return res
# fmt: on

# fmt: off
@njit(nogil=True, parallel=True, cache=True)
def eval_loads_2d(
    size: tuple,
    shape: tuple, 
    coeffs:ndarray, 
    points:ndarray,
    ) -> ndarray:
    """
    Evaluates the loads at some points for plate problems.
    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nDOF = 3
    res = np.zeros((nP, nDOF), dtype=float)
    for iP in prange(nP):
        xP, yP = points[iP]
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iP, 0] += (
                    coeffs[iMN, 0]
                    * np.sin(xP * np.pi * m / Lx) 
                    * np.sin(yP * np.pi * n / Ly)
                )
                res[iP, 1] += (
                    coeffs[iMN, 1]
                    * np.sin(xP * np.pi * m / Lx) 
                    * np.cos(yP * np.pi * n / Ly)
                )
                res[iP, 2] += (
                    coeffs[iMN, 2]
                    * np.cos(xP * np.pi * m / Lx) 
                    * np.sin(yP * np.pi * n / Ly)
                )
    return res
# fmt: on
