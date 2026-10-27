from types import NoneType
from typing import Iterable

import numpy as np
from numba import njit, prange
from numpy import cos, ndarray, sin
from numpy import pi as PI
from sigmaepsilon.math import atleast1d, atleast2d, atleast3d, atleast4d, itype_of_ftype

from .config import Config
from .enums import PostprocMethod
from .postproc_gemm import postproc_Kirchhoff_gemm, postproc_Mindlin_gemm

__all__ = ["postproc", "postproc_Kirchhoff", "postproc_Mindlin"]


UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(range(13))


def postproc(
    size: float | Iterable[float],
    shape: int | Iterable[int],
    points: ndarray,
    solution: ndarray,
    loads: ndarray,
    D: float | ndarray,
    S: float | ndarray | NoneType = None,
    loads0: ndarray | NoneType = None,
) -> ndarray:
    """
    Calculates postprocessing items.

    If the zero-order load coefficients `loads0` are provided (see
    :func:`~sigmaepsilon.solid.fourier.loads.LoadCase.rhs0`), the contributions
    of the corresponding harmonics are added to the results.

    Plates are postprocessed with dense matrix products (see :mod:`.postproc_gemm`)
    or with the Numba kernels, depending on `Config.POSTPROC_METHOD`. Beams always
    use the Numba kernels.
    """
    ftype = solution.dtype
    itype = itype_of_ftype(ftype)
    if PostprocMethod.from_value(Config.POSTPROC_METHOD) is PostprocMethod.BLAS:
        mindlin, kirchhoff = postproc_Mindlin_gemm, postproc_Kirchhoff_gemm
    else:
        mindlin, kirchhoff = postproc_Mindlin, postproc_Kirchhoff
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
            res = mindlin(
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
            res = kirchhoff(
                np.array(size).astype(ftype),
                np.array(shape).astype(itype),
                atleast2d(points).astype(ftype),
                atleast3d(solution).astype(ftype),
                atleast3d(D),
                loads,
            )
    if loads0 is not None:
        res += postproc_zero_order(size, shape, points, res.shape[0], loads0, D, S)

    return res


def postproc_zero_order(
    size: float | Iterable[float],
    shape: int | Iterable[int],
    points: ndarray,
    nLHS: int,
    loads0: ndarray,
    D: float | ndarray,
    S: float | ndarray | NoneType = None,
) -> ndarray:
    """
    Calculates the contributions of the zero-order harmonics of moment loads
    to the postprocessing items. These harmonics have no transverse displacement
    and a single unknown each, hence they are solved here in closed form.

    Returns
    -------
    numpy.ndarray
        4d array of shape (nLHS, nRHS, nP, ...).

    """
    ftype = loads0.dtype
    itype = itype_of_ftype(ftype)
    if not isinstance(size, Iterable):
        points = atleast1d(points).astype(ftype)
        loads0 = atleast2d(loads0)
        if S is not None:
            return postproc_Timoshenko_zero_order(points, loads0, atleast1d(S))
        else:
            return postproc_Bernoulli_zero_order(nLHS, points, loads0)
    else:
        size = np.array(size).astype(ftype)
        shape = np.array(shape).astype(itype)
        points = atleast2d(points).astype(ftype)
        loads0 = atleast3d(loads0)
        if S is not None:
            return postproc_Mindlin_zero_order(
                size, shape, points, loads0, atleast3d(D), atleast3d(S)
            )
        else:
            return postproc_Kirchhoff_zero_order(nLHS, size, shape, points, loads0)


@njit(nogil=True, parallel=True, cache=True)
def postproc_Bernoulli_zero_order(
    nLHS: int, points: ndarray, loads0: ndarray
) -> ndarray:
    """
    Contributions of the mean of the moment loads for Euler-Bernoulli beams,
    as a 4d array of shape (nLHS, nRHS, nP, 6). `loads0` is a 2d array of shape
    (nRHS, 2).
    """
    nP = points.shape[0]
    nRHS = loads0.shape[0]
    res = np.zeros((nLHS, nRHS, nP, 6), dtype=loads0.dtype)
    for iP in prange(nP):
        for iRHS in range(nRHS):
            r0 = loads0[iRHS, 1]
            for iLHS in range(nLHS):
                res[iLHS, iRHS, iP, 5] = -r0
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Timoshenko_zero_order(
    points: ndarray, loads0: ndarray, GA: ndarray
) -> ndarray:
    """
    Contributions of the mean of the moment loads for Timoshenko-Ehrenfest beams,
    as a 4d array of shape (nLHS, nRHS, nP, 6). `loads0` is a 2d array of shape
    (nRHS, 2).
    """
    nP = points.shape[0]
    nRHS = loads0.shape[0]
    nLHS = GA.shape[0]
    res = np.zeros((nLHS, nRHS, nP, 6), dtype=loads0.dtype)
    for iP in prange(nP):
        for iRHS in range(nRHS):
            r0 = loads0[iRHS, 1]
            for iLHS in range(nLHS):
                rot0 = r0 / GA[iLHS]
                res[iLHS, iRHS, iP, 1] = rot0
                res[iLHS, iRHS, iP, 3] = -rot0
                res[iLHS, iRHS, iP, 5] = -r0
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Kirchhoff_zero_order(
    nLHS: int, size: ndarray, shape: ndarray, points: ndarray, loads0: ndarray
) -> ndarray:
    """
    Contributions of the harmonics (m, 0) of mx and (0, n) of my for Kirchhoff-Love
    plates, as a 4d array of shape (nLHS, nRHS, nP, 13). `loads0` is a 3d array of
    shape (nRHS, M + N, 3).
    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nRHS = loads0.shape[0]
    res = np.zeros((nLHS, nRHS, nP, 13), dtype=loads0.dtype)
    for iP in prange(nP):
        x, y = points[iP, :2]
        for iRHS in range(nRHS):
            qx, qy = 0.0, 0.0
            for m in range(1, M + 1):
                qy -= loads0[iRHS, m - 1, 1] * sin(PI * m * x / Lx)
            for n in range(1, N + 1):
                qx += loads0[iRHS, M + n - 1, 2] * sin(PI * n * y / Ly)
            for iLHS in range(nLHS):
                res[iLHS, iRHS, iP, QX] = qx
                res[iLHS, iRHS, iP, QY] = qy
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Mindlin_zero_order(
    size: ndarray,
    shape: ndarray,
    points: ndarray,
    loads0: ndarray,
    D: ndarray,
    S: ndarray,
) -> ndarray:
    """
    Contributions of the harmonics (m, 0) of mx and (0, n) of my for Uflyand-Mindlin
    plates, as a 4d array of shape (nLHS, nRHS, nP, 13). `loads0` is a 3d array of
    shape (nRHS, M + N, 3).
    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nRHS = loads0.shape[0]
    nLHS = D.shape[0]
    res = np.zeros((nLHS, nRHS, nP, 13), dtype=loads0.dtype)
    for iP in prange(nP):
        x, y = points[iP, :2]
        for iRHS in range(nRHS):
            for iLHS in range(nLHS):
                D66 = D[iLHS, 2, 2]
                S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
                # harmonics (m, 0): only the rotation x is non-zero
                for m in range(1, M + 1):
                    Bm0 = loads0[iRHS, m - 1, 1] / (PI**2 * D66 * m**2 / Lx**2 + S44)
                    Sm = sin(PI * m * x / Lx)
                    Cm = cos(PI * m * x / Lx)
                    res[iLHS, iRHS, iP, ROTX] += Bm0 * Sm
                    res[iLHS, iRHS, iP, CXY] += -PI * Bm0 * m * Cm / Lx
                    res[iLHS, iRHS, iP, EYZ] += -Bm0 * Sm
                    res[iLHS, iRHS, iP, MXY] += -PI * Bm0 * D66 * m * Cm / Lx
                    res[iLHS, iRHS, iP, QY] += -Bm0 * S44 * Sm
                # harmonics (0, n): only the rotation y is non-zero
                for n in range(1, N + 1):
                    C0n = loads0[iRHS, M + n - 1, 2] / (
                        PI**2 * D66 * n**2 / Ly**2 + S55
                    )
                    Sn = sin(PI * n * y / Ly)
                    Cn = cos(PI * n * y / Ly)
                    res[iLHS, iRHS, iP, ROTY] += C0n * Sn
                    res[iLHS, iRHS, iP, CXY] += PI * C0n * n * Cn / Ly
                    res[iLHS, iRHS, iP, EXZ] += C0n * Sn
                    res[iLHS, iRHS, iP, MXY] += PI * C0n * D66 * n * Cn / Ly
                    res[iLHS, iRHS, iP, QX] += C0n * S55 * Sn
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
        4d array of shape (nLHS, nRHS, nP, ...) of post-processing items.
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
    res = np.zeros((nLHS, nRHS, nP, 6), dtype=solution.dtype)
    for iP in prange(nP):
        x = points[iP]
        for n in range(1, N + 1):
            iN = n - 1
            arg = PI * n / L
            Sn = sin(x * arg)
            Cn = cos(x * arg)
            for iLHS in range(nLHS):
                for iRHS in range(nRHS):
                    vn = solution[iLHS, iRHS, iN]
                    q = loads[iRHS, iN, 1]
                    res[iLHS, iRHS, iP, 0] += vn * Sn
                    res[iLHS, iRHS, iP, 1] += vn * arg * Cn
                    res[iLHS, iRHS, iP, 2] += -vn * arg**2 * Sn
                    res[iLHS, iRHS, iP, 4] += -EI[iLHS] * vn * arg**2 * Sn
                    res[iLHS, iRHS, iP, 5] += (EI[iLHS] * vn * arg**3 - q) * Cn
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
        4d array of shape (nLHS, nRHS, nP, ...) of post-processing items.
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
    res = np.zeros((nLHS, nRHS, nP, 6), dtype=solution.dtype)
    for iP in prange(nP):
        x = points[iP]
        for n in range(1, N + 1):
            iN = n - 1
            arg = PI * n / L
            Sn = sin(x * arg)
            Cn = cos(x * arg)
            for iLHS in range(nLHS):
                for iRHS in range(nRHS):
                    vn, rn = solution[iLHS, iRHS, iN]
                    res[iLHS, iRHS, iP, 0] += vn * Sn
                    res[iLHS, iRHS, iP, 1] += rn * Cn
                    res[iLHS, iRHS, iP, 2] += -rn * arg * Sn
                    res[iLHS, iRHS, iP, 3] += (vn * arg - rn) * Cn
                    res[iLHS, iRHS, iP, 4] += -EI[iLHS] * rn * arg * Sn
                    res[iLHS, iRHS, iP, 5] += GA[iLHS] * (vn * arg - rn) * Cn
    return res


@njit(nogil=True, parallel=True, cache=True)
def postproc_Mindlin(
    size: ndarray,
    shape: ndarray,
    points: ndarray,
    solution: ndarray,
    D: ndarray,
    S: ndarray,
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
        4d array of shape (nLHS, nRHS, nP, ...) of post-processing items.
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
    res = np.zeros((nLHS, nRHS, nP, 13), dtype=D.dtype)
    for iP in prange(nP):
        xp, yp = points[iP, :2]
        Sy = np.empty(N, dtype=D.dtype)
        Cy = np.empty(N, dtype=D.dtype)
        for n in range(1, N + 1):
            Sy[n - 1] = sin(PI * n * yp / Ly)
            Cy[n - 1] = cos(PI * n * yp / Ly)
        for m in range(1, M + 1):
            Sm = sin(PI * m * xp / Lx)
            Cm = cos(PI * m * xp / Lx)
            for n in range(1, N + 1):
                iMN = (m - 1) * N + n - 1
                Sn = Sy[n - 1]
                Cn = Cy[n - 1]
                for iLHS in range(nLHS):
                    D11, D12, D22, D66 = (
                        D[iLHS, 0, 0],
                        D[iLHS, 0, 1],
                        D[iLHS, 1, 1],
                        D[iLHS, 2, 2],
                    )
                    S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
                    for iRHS in range(nRHS):
                        Amn, Bmn, Cmn = solution[iLHS, iRHS, iMN]
                        r = res[iLHS, iRHS, iP]
                        r[UZ] += Amn * Sm * Sn
                        r[ROTX] += Bmn * Sm * Cn
                        r[ROTY] += Cmn * Sn * Cm
                        r[CX] += -PI * Cmn * m * Sm * Sn / Lx
                        r[CY] += PI * Bmn * n * Sm * Sn / Ly
                        r[CXY] += (
                            -PI * Bmn * m * Cm * Cn / Lx + PI * Cmn * n * Cm * Cn / Ly
                        )
                        r[EXZ] += Cmn * Sn * Cm + PI * Amn * m * Sn * Cm / Lx
                        r[EYZ] += -Bmn * Sm * Cn + PI * Amn * n * Sm * Cn / Ly
                        r[MX] += (
                            PI * Bmn * D12 * n * Sm * Sn / Ly
                            - PI * Cmn * D11 * m * Sm * Sn / Lx
                        )
                        r[MY] += (
                            PI * Bmn * D22 * n * Sm * Sn / Ly
                            - PI * Cmn * D12 * m * Sm * Sn / Lx
                        )
                        r[MXY] += (
                            -PI * Bmn * D66 * m * Cm * Cn / Lx
                            + PI * Cmn * D66 * n * Cm * Cn / Ly
                        )
                        r[QX] += Cmn * S55 * Sn * Cm + PI * Amn * S55 * m * Sn * Cm / Lx
                        r[QY] += (
                            -Bmn * S44 * Sm * Cn + PI * Amn * S44 * n * Sm * Cn / Ly
                        )
    return res


# fmt: off
@njit(nogil=True, parallel=True, cache=True)
def postproc_Kirchhoff(
    size: ndarray, shape: ndarray, points: ndarray, solution: ndarray, D: ndarray, loads: ndarray
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
    numpy.ndarray
        4d array of shape (nLHS, nRHS, nP, ...) of post-processing items. The indices along
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
    res = np.zeros((nLHS, nRHS, nP, 13), dtype=D.dtype)
    for iP in prange(nP):
        x, y = points[iP, :2]
        Sy = np.empty(N, dtype=D.dtype)
        Cy = np.empty(N, dtype=D.dtype)
        for n in range(1, N + 1):
            Sy[n - 1] = sin(PI * n * y / Ly)
            Cy[n - 1] = cos(PI * n * y / Ly)
        for m in range(1, M + 1):
            Sm = sin(PI * m * x / Lx)
            Cm = cos(PI * m * x / Lx)
            for n in range(1, N + 1):
                Sn = Sy[n - 1]
                Cn = Cy[n - 1]
                iMN = (m - 1) * N + n - 1
                for iLHS in range(nLHS):
                    D11, D12, D22, D66 = (
                        D[iLHS, 0, 0],
                        D[iLHS, 0, 1],
                        D[iLHS, 1, 1],
                        D[iLHS, 2, 2],
                    )
                    for iRHS in range(nRHS):
                        Amn = solution[iLHS, iRHS, iMN]
                        qmn, rmn = loads[iRHS, iMN, 1:]
                        r = res[iLHS, iRHS, iP]
                        r[UZ] += Amn * Sm * Sn
                        r[ROTX] += PI * Amn * n * Sm * Cn / Ly
                        r[ROTY] += -PI * Amn * m * Sn * Cm / Lx
                        r[CX] += PI**2 * Amn * m**2 * Sm * Sn / Lx**2
                        r[CY] += PI**2 * Amn * n**2 * Sm * Sn / Ly**2
                        r[CXY] += -2 * PI**2 * Amn * m * n * Cm * Cn / (Lx * Ly)
                        r[MX] += (
                            PI**2 * Amn * D11 * m**2 * Sm * Sn / Lx**2
                            + PI**2 * Amn * D12 * n**2 * Sm * Sn / Ly**2
                        )
                        r[MY] += (
                            PI**2 * Amn * D12 * m**2 * Sm * Sn / Lx**2
                            + PI**2 * Amn * D22 * n**2 * Sm * Sn / Ly**2
                        )
                        r[MXY] += -2 * PI**2 * Amn * D66 * m * n * Cm * Cn / (Lx * Ly)
                        r[QX] += (
                            PI**3 * Amn * D11 * m**3 * Sn * Cm / Lx**3
                            + PI**3 * Amn * D12 * m * n**2 * Sn * Cm / (Lx * Ly**2)
                            + 2 * PI**3 * Amn * D66 * m * n**2 * Sn * Cm / (Lx * Ly**2)
                            + rmn * Sn * Cm
                        )
                        r[QY] += (
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
