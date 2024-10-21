from typing import Iterable, Callable
from types import NoneType

import numpy as np
from numba import njit, prange
from numpy import ndarray, ndarray

from sigmaepsilon.math.function import Function

from .config import config


@njit(nogil=True, parallel=True, cache=True)
def _monte_carlo_1d_njit(
    size: float,
    shape: int,
    points: ndarray,
    fvalues: ndarray,
    dsize: float,
    num_MC_tot: int,
    out: ndarray,
) -> ndarray:
    N = shape
    L = size
    multi = (2 / L) * dsize
    factor = np.pi / L

    for n in prange(1, N + 1):
        iN = n - 1
        out[0, iN, 0] += (
            multi * np.sum(np.sin(points * factor * n) * fvalues[:, 0]) / num_MC_tot
        )
        out[0, iN, 1] += (
            multi * np.sum(np.cos(points * factor * n) * fvalues[:, 1]) / num_MC_tot
        )

    return out


def _monte_carlo_1d(
    size: float,
    shape: int,
    values: Iterable,
    dsize: float,
    *,
    rpg: Callable[[int], ndarray],
    out: ndarray | NoneType = None,
) -> ndarray:
    N = shape

    if out is None:
        out = np.zeros((1, N, 2), dtype=float)

    functions = []
    for i in range(2):
        if isinstance(values[i], str):
            function = Function(values[i], variables=["x"])
        elif isinstance(values[i], (float, int)):
            function = lambda points: np.full(
                (len(points),), values[i], dtype=float
            )
        else:  # pragma: no cover
            raise ValueError(f"Invalid value {values[i]}")
        functions.append(function)

    num_MC_samples = config.get("NUM_MC_SAMPLES_BEAM")
    MC_batch_size = config.get("MC_BATCH_SIZE_BEAM")
    num_MC_bacthes = int(num_MC_samples // MC_batch_size)
    remaining_MC_points = int(num_MC_samples % MC_batch_size)
    f_vals = np.zeros((MC_batch_size, 2), dtype=float)
    points = np.zeros((MC_batch_size), dtype=float)

    for _ in range(num_MC_bacthes):
        rpg(MC_batch_size, out=points)
        for i in range(2):
            f_vals[:, i] = functions[i]([points])
        _monte_carlo_1d_njit(size, shape, points, f_vals, dsize, num_MC_samples, out)

    MC_batch_size = remaining_MC_points
    if remaining_MC_points > 1:
        points = points[:MC_batch_size]
        f_vals = f_vals[:MC_batch_size]
        rpg(MC_batch_size, out=points)
        for i in range(2):
            f_vals[:, i] = functions[i]([points])
        _monte_carlo_1d_njit(size, shape, points, f_vals, dsize, num_MC_samples, out)

    return out


@njit(nogil=True, parallel=True, cache=True)
def _monte_carlo_2d_njit(
    size: tuple,
    shape: tuple,
    points: ndarray,
    fvalues: ndarray,
    dsize: float,
    num_MC_tot: int,
    out: ndarray,
) -> ndarray:
    M, N = shape
    Lx, Ly = size
    multi = (4 / Lx / Ly) * dsize
    px = points[:, 0]
    py = points[:, 1]
    factor_x = np.pi / Lx
    factor_y = np.pi / Ly

    for m in prange(1, M + 1):
        for n in prange(1, N + 1):
            mn = (m - 1) * N + n - 1
            out[0, mn, 0] += (
                multi
                * np.sum(
                    np.sin(px * factor_x * m)
                    * np.sin(py * factor_y * n)
                    * fvalues[:, 0]
                )
                / num_MC_tot
            )
            out[0, mn, 1] += (
                multi
                * np.sum(
                    np.sin(px * factor_x * m)
                    * np.cos(py * factor_y * n)
                    * fvalues[:, 1]
                )
                / num_MC_tot
            )
            out[0, mn, 2] += (
                multi
                * np.sum(
                    np.cos(px * factor_x * m)
                    * np.sin(py * factor_y * n)
                    * fvalues[:, 2]
                )
                / num_MC_tot
            )

    return out


def _monte_carlo_2d(
    size: tuple,
    shape: tuple,
    values: Iterable,
    dsize: float,
    *,
    rpg: Callable[[int], ndarray],
    out: ndarray | NoneType = None,
) -> ndarray:
    M, N = shape

    if out is None:
        out = np.zeros((1, M * N, 3), dtype=float)

    functions = []
    for i in range(3):
        if isinstance(values[i], str):
            function = Function(values[i], variables=["x", "y"])
        elif isinstance(values[i], (float, int)):
            function = lambda points: np.full(
                (points.shape[1],), values[i], dtype=float
            )
        else:  # pragma: no cover
            raise ValueError(f"Invalid value {values[i]}")
        functions.append(function)

    num_MC_samples = config.get("NUM_MC_SAMPLES_PLATE")
    MC_batch_size = config.get("MC_BATCH_SIZE_PLATE")
    num_MC_bacthes = int(num_MC_samples // MC_batch_size)
    remaining_MC_points = int(num_MC_samples % MC_batch_size)
    f_vals = np.zeros((MC_batch_size, 3), dtype=float)
    points = np.zeros((MC_batch_size, 2), dtype=float)

    for _ in range(num_MC_bacthes):
        rpg(MC_batch_size, out=points)
        for i in range(3):
            f_vals[:, i] = functions[i](points.T)
        _monte_carlo_2d_njit(size, shape, points, f_vals, dsize, num_MC_samples, out)

    MC_batch_size = remaining_MC_points
    if remaining_MC_points > 1:
        points = points[:MC_batch_size]
        f_vals = f_vals[:MC_batch_size]
        rpg(MC_batch_size, out=points)
        for i in range(3):
            f_vals[:, i] = functions[i](points.T)
        _monte_carlo_2d_njit(size, shape, points, f_vals, dsize, num_MC_samples, out)

    return out
