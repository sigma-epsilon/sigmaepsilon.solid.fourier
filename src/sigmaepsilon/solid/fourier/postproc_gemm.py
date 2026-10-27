"""
Postprocessing of plates with dense matrix products.

Every postprocessing quantity of a plate is a double sum over the harmonics
with one of four products of basis functions,

.. math::

    \\sin(\\alpha_m x) \\sin(\\beta_n y), \\quad
    \\sin(\\alpha_m x) \\cos(\\beta_n y), \\quad
    \\cos(\\alpha_m x) \\sin(\\beta_n y), \\quad
    \\cos(\\alpha_m x) \\cos(\\beta_n y),

where :math:`\\alpha_m = \\pi m / L_x` and :math:`\\beta_n = \\pi n / L_y`. Hence each
quantity can be written as :math:`X(x)^T K Y(y)`, where :math:`X` and :math:`Y` are
the basis vectors of length M and N, and :math:`K` is an M x N matrix of
coefficients that folds in the solution amplitudes and the stiffnesses.

The quantities that share the same basis products are evaluated together, with
one matrix product per group. If the points form a (possibly partial) tensor
product grid, the results are evaluated on the grid as :math:`X K Y^T`, which is
much cheaper than evaluating them point by point.
"""

import numpy as np
from numpy import ndarray
from numpy import pi as PI

from .config import Config

__all__ = ["postproc_Kirchhoff_gemm", "postproc_Mindlin_gemm"]


UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(range(13))
NUM_COMPONENTS = 13

SIN, COS = 0, 1


# COEFFICIENT MATRICES


def _wave_numbers(size: ndarray, shape: ndarray) -> tuple[ndarray, ndarray]:
    Lx, Ly = size
    M, N = shape
    a = PI * np.arange(1, M + 1) / Lx
    b = PI * np.arange(1, N + 1) / Ly
    return a, b


def _per_lhs(values: ndarray) -> ndarray:
    # values of shape (nLHS,) broadcast against arrays of shape (nLHS, nRHS, M, N)
    return values[:, None, None, None]


def _mindlin_groups(
    size: ndarray, shape: ndarray, solution: ndarray, D: ndarray, S: ndarray
) -> list[tuple[tuple[int, int], dict[int, ndarray]]]:
    M, N = shape
    nLHS, nRHS = solution.shape[:2]
    sol = solution.reshape(nLHS, nRHS, M, N, 3)
    A, B, C = sol[..., 0], sol[..., 1], sol[..., 2]
    a, b = _wave_numbers(size, shape)
    a, b = a[:, None], b[None, :]
    D11, D12, D22, D66 = (
        _per_lhs(D[:, i, j]) for i, j in [(0, 0), (0, 1), (1, 1), (2, 2)]
    )
    S44, S55 = _per_lhs(S[:, 0, 0]), _per_lhs(S[:, 1, 1])
    exz, eyz, cxy = C + A * a, -B + A * b, -B * a + C * b
    return [
        (
            (SIN, SIN),
            {
                UZ: A,
                CX: -C * a,
                CY: B * b,
                MX: D12 * B * b - D11 * C * a,
                MY: D22 * B * b - D12 * C * a,
            },
        ),
        ((SIN, COS), {ROTX: B, EYZ: eyz, QY: S44 * eyz}),
        ((COS, SIN), {ROTY: C, EXZ: exz, QX: S55 * exz}),
        ((COS, COS), {CXY: cxy, MXY: D66 * cxy}),
    ]


def _kirchhoff_groups(
    size: ndarray, shape: ndarray, solution: ndarray, D: ndarray, loads: ndarray
) -> list[tuple[tuple[int, int], dict[int, ndarray]]]:
    M, N = shape
    nLHS, nRHS = solution.shape[:2]
    A = solution.reshape(nLHS, nRHS, M, N)
    # the moment loads contribute to the shear forces directly
    q = loads[:, :, 1].reshape(1, nRHS, M, N)
    r = loads[:, :, 2].reshape(1, nRHS, M, N)
    a, b = _wave_numbers(size, shape)
    a, b = a[:, None], b[None, :]
    D11, D12, D22, D66 = (
        _per_lhs(D[:, i, j]) for i, j in [(0, 0), (0, 1), (1, 1), (2, 2)]
    )
    cxy = -2 * A * a * b
    return [
        (
            (SIN, SIN),
            {
                UZ: A,
                CX: A * a**2,
                CY: A * b**2,
                MX: A * (D11 * a**2 + D12 * b**2),
                MY: A * (D12 * a**2 + D22 * b**2),
            },
        ),
        (
            (SIN, COS),
            {
                ROTX: A * b,
                QY: A * (D12 * a**2 * b + D22 * b**3 + 2 * D66 * a**2 * b) - q,
            },
        ),
        (
            (COS, SIN),
            {
                ROTY: -A * a,
                QX: A * (D11 * a**3 + D12 * a * b**2 + 2 * D66 * a * b**2) + r,
            },
        ),
        ((COS, COS), {CXY: cxy, MXY: D66 * cxy}),
    ]


def _stack_groups(
    groups: list[tuple[tuple[int, int], dict[int, ndarray]]], nLHS: int, nRHS: int
) -> list[tuple[tuple[int, int], list[int], ndarray]]:
    # The coefficient matrices of a group are stacked side by side into a matrix
    # of shape (M, k * nLHS * nRHS * N), so that a single matrix product evaluates
    # every component of the group, for all left- and right-hand sides.
    stacked = []
    for basis, components in groups:
        indices = list(components)
        K = np.stack(
            [
                np.broadcast_to(components[i], (nLHS, nRHS) + components[i].shape[2:])
                for i in indices
            ]
        )
        # K.shape = (k, nLHS, nRHS, M, N)
        M = K.shape[3]
        K = np.ascontiguousarray(K.transpose(3, 0, 1, 2, 4)).reshape(M, -1)
        stacked.append((basis, indices, K))
    return stacked


# EVALUATION


def _basis(coords: ndarray, wave_numbers: ndarray) -> tuple[ndarray, ndarray]:
    arg = np.outer(coords, wave_numbers)
    return np.sin(arg), np.cos(arg)


def _use_grid(nP: int, nx: int, ny: int, M: int, N: int) -> bool:
    # floating point operations per column of the stacked coefficient matrices
    scattered = nP * (M * N + N)
    grid = nx * M * N + nx * ny * N
    return grid < scattered


def _evaluate_scattered(
    size: ndarray,
    shape: ndarray,
    points: ndarray,
    stacked: list,
    res: ndarray,
) -> None:
    M, N = shape
    nLHS, nRHS, nP = res.shape[:3]
    a, b = _wave_numbers(size, shape)
    width = max(K.shape[1] for _, _, K in stacked)
    batch = max(1, Config.POSTPROC_BUFFER_SIZE // (width + 2 * (M + N)))
    for start in range(0, nP, batch):
        stop = min(start + batch, nP)
        X = _basis(points[start:stop, 0], a)
        Y = _basis(points[start:stop, 1], b)
        for (bx, by), indices, K in stacked:
            T = (X[bx] @ K).reshape(stop - start, -1, N)
            values = np.einsum("pwn,pn->pw", T, Y[by])
            values = values.reshape(stop - start, len(indices), nLHS, nRHS)
            res[:, :, start:stop, indices] = values.transpose(2, 3, 0, 1)


def _evaluate_grid(
    size: ndarray,
    shape: ndarray,
    x: ndarray,
    ix: ndarray,
    y: ndarray,
    iy: ndarray,
    stacked: list,
    res: ndarray,
) -> None:
    # x and y are the coordinates of the grid lines, ix and iy are the indices
    # of the grid lines of the points
    N = shape[1]
    nLHS, nRHS = res.shape[:2]
    nx, ny = len(x), len(y)
    a, b = _wave_numbers(size, shape)
    X = _basis(x, a)
    Y = _basis(y, b)
    width = max(K.shape[1] for _, _, K in stacked)
    # the grid is processed in strips of rows, to limit the memory footprint
    rows = max(1, Config.POSTPROC_BUFFER_SIZE // (width // N * (N + ny)))
    for start in range(0, nx, rows):
        stop = min(start + rows, nx)
        # every grid line has at least one point, so the selection is never empty
        selection = np.nonzero((ix >= start) & (ix < stop))[0]
        jx, jy = ix[selection] - start, iy[selection]
        for (bx, by), indices, K in stacked:
            T = (X[bx][start:stop] @ K).reshape(-1, N)
            values = (T @ Y[by].T).reshape(stop - start, -1, ny)[jx, :, jy]
            values = values.reshape(len(selection), len(indices), nLHS, nRHS)
            res[:, :, selection[:, None], indices] = values.transpose(2, 3, 0, 1)


def _evaluate(
    size: ndarray,
    shape: ndarray,
    points: ndarray,
    groups: list,
    nLHS: int,
    nRHS: int,
    grid: bool | None,
    dtype: np.dtype,
) -> ndarray:
    M, N = shape
    nP = points.shape[0]
    res = np.zeros((nLHS, nRHS, nP, NUM_COMPONENTS), dtype=dtype)
    if nP == 0:
        return res
    stacked = _stack_groups(groups, nLHS, nRHS)
    if grid is False:
        _evaluate_scattered(size, shape, points, stacked, res)
        return res
    x, ix = np.unique(points[:, 0], return_inverse=True)
    y, iy = np.unique(points[:, 1], return_inverse=True)
    if grid or _use_grid(nP, len(x), len(y), M, N):
        _evaluate_grid(size, shape, x, ix.ravel(), y, iy.ravel(), stacked, res)
    else:
        _evaluate_scattered(size, shape, points, stacked, res)
    return res


def postproc_Mindlin_gemm(
    size: ndarray,
    shape: ndarray,
    points: ndarray,
    solution: ndarray,
    D: ndarray,
    S: ndarray,
    grid: bool | None = None,
) -> ndarray:
    """
    Calculates postprocessing quantities of Uflyand-Mindlin plates at selected
    points for multiple left- and right-hand sides, using matrix products.
    The arguments and the returned array are the same as for
    :func:`~sigmaepsilon.solid.fourier.postproc.postproc_Mindlin`.

    Parameters
    ----------
    grid: bool, Optional
        If True, the results are evaluated on the tensor product grid spanned
        by the coordinates of the points. If False, they are evaluated point by
        point. If None, the cheaper of the two is chosen. Default is None.
    """
    nLHS, nRHS = solution.shape[:2]
    groups = _mindlin_groups(size, shape, solution, D, S)
    return _evaluate(size, shape, points, groups, nLHS, nRHS, grid, solution.dtype)


def postproc_Kirchhoff_gemm(
    size: ndarray,
    shape: ndarray,
    points: ndarray,
    solution: ndarray,
    D: ndarray,
    loads: ndarray,
    grid: bool | None = None,
) -> ndarray:
    """
    Calculates postprocessing quantities of Kirchhoff-Love plates at selected
    points for multiple left- and right-hand sides, using matrix products.
    The arguments and the returned array are the same as for
    :func:`~sigmaepsilon.solid.fourier.postproc.postproc_Kirchhoff`.

    Parameters
    ----------
    grid: bool, Optional
        If True, the results are evaluated on the tensor product grid spanned
        by the coordinates of the points. If False, they are evaluated point by
        point. If None, the cheaper of the two is chosen. Default is None.
    """
    nLHS, nRHS = solution.shape[:2]
    groups = _kirchhoff_groups(size, shape, solution, D, loads)
    return _evaluate(size, shape, points, groups, nLHS, nRHS, grid, solution.dtype)
