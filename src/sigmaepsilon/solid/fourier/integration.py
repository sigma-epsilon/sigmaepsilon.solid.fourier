"""
Numerical integration of the Fourier coefficients of loads whose values are
arbitrary expressions, or whose domain has no closed-form coefficients.

The integrals are evaluated as weighted sums over integration points, either
with deterministic Gauss-Legendre quadrature (see :mod:`.quadrature`) or with
Monte Carlo sampling (see :mod:`.mc`). The projections onto the Fourier basis
are matrix products, so that the number of trigonometric evaluations grows
with (M + N) times the number of points, instead of M * N times.

Every function returns the coefficients in the layout of
:func:`~sigmaepsilon.solid.fourier.loads.LoadCase.rhs`, or with ``zero_order=True``
in the layout of :func:`~sigmaepsilon.solid.fourier.loads.LoadCase.rhs0`.
"""

from collections.abc import Callable, Iterable, Iterator
from functools import partial
from numbers import Real

import numpy as np
from numpy import ndarray
from sigmaepsilon.math.function import Function

from .config import Config
from .enums import IntegrationMethod
from .mc import monte_carlo_samples
from .quadrature import disk_rule, line_1d_rule, line_2d_rule, rectangle_rule
from .utils import (
    RNGType,
    generate_random_points_in_disk,
    generate_random_points_in_rectangle,
    generate_random_points_on_line_segment_1d,
    generate_random_points_on_line_segment_2d,
)

__all__ = [
    "integrate_disk",
    "integrate_line_1d",
    "integrate_line_2d",
    "integrate_rectangle",
]


# LOAD FUNCTIONS


def _load_functions(values: Iterable, variables: list[str]) -> list[float | Function]:
    functions = []
    for value in values:
        if isinstance(value, str):
            functions.append(Function(value, variables=variables))
        elif isinstance(value, Real) and not isinstance(value, bool):
            functions.append(float(value))
        else:
            raise TypeError(f"Invalid load value {value!r}.")
    return functions


def _eval_load_functions(
    functions: list[float | Function], coords: list[ndarray]
) -> ndarray:
    # coords is a list of coordinate arrays of the same shape, one per variable
    res = np.zeros(coords[0].shape + (len(functions),), dtype=float)
    for i, function in enumerate(functions):
        if isinstance(function, float):
            res[..., i] = function
        else:
            res[..., i] = function(coords)
    return res


def _has_no_moment_load(values: Iterable) -> bool:
    return all(isinstance(v, Real) and v == 0 for v in values[1:])


# PROJECTIONS


def _project_1d(
    L: float,
    N: int,
    x: ndarray,
    w: ndarray,
    f: ndarray,
    out: ndarray,
    zero_order: bool,
) -> None:
    if zero_order:
        # the mean of the moment load
        out[1] += np.dot(w, f[:, 1]) / L
        return
    arg = np.outer(x, np.arange(1, N + 1) * (np.pi / L))
    out[:, 0] += (2 / L) * (np.sin(arg).T @ (w * f[:, 0]))
    out[:, 1] += (2 / L) * (np.cos(arg).T @ (w * f[:, 1]))


def _project_2d(
    size: tuple,
    shape: tuple,
    xy: ndarray,
    w: ndarray,
    f: ndarray,
    out: ndarray,
    zero_order: bool,
) -> None:
    Lx, Ly = size
    M, N = shape
    ax = np.outer(xy[:, 0], np.arange(1, M + 1) * (np.pi / Lx))
    ay = np.outer(xy[:, 1], np.arange(1, N + 1) * (np.pi / Ly))
    wf = w[:, None] * f
    if zero_order:
        # mx at the harmonics (m, 0) and my at the harmonics (0, n)
        c = 2 / (Lx * Ly)
        out[:M, 1] += c * (np.sin(ax).T @ wf[:, 1])
        out[M:, 2] += c * (np.sin(ay).T @ wf[:, 2])
        return
    c = 4 / (Lx * Ly)
    Sx, Sy = np.sin(ax), np.sin(ay)
    out[:, 0] += c * ((Sx * wf[:, :1]).T @ Sy).ravel()
    out[:, 1] += c * ((Sx * wf[:, 1:2]).T @ np.cos(ay)).ravel()
    out[:, 2] += c * ((np.cos(ax) * wf[:, 2:3]).T @ Sy).ravel()


def _project_2d_tensor(
    size: tuple,
    shape: tuple,
    x: ndarray,
    y: ndarray,
    G: ndarray,
    out: ndarray,
    zero_order: bool,
) -> None:
    # G[i, j, k] is the weighted value of the k-th load component at (x[i], y[j]),
    # the projection is separable, F_k = Bx_k^T @ G_k @ By_k
    Lx, Ly = size
    M, N = shape
    ax = np.outer(x, np.arange(1, M + 1) * (np.pi / Lx))
    ay = np.outer(y, np.arange(1, N + 1) * (np.pi / Ly))
    if zero_order:
        c = 2 / (Lx * Ly)
        out[:M, 1] += c * (np.sin(ax).T @ G[:, :, 1].sum(axis=1))
        out[M:, 2] += c * (np.sin(ay).T @ G[:, :, 2].sum(axis=0))
        return
    c = 4 / (Lx * Ly)
    Sx, Sy = np.sin(ax), np.sin(ay)
    out[:, 0] += c * (Sx.T @ G[:, :, 0] @ Sy).ravel()
    out[:, 1] += c * (Sx.T @ G[:, :, 1] @ np.cos(ay)).ravel()
    out[:, 2] += c * (np.cos(ax).T @ G[:, :, 2] @ Sy).ravel()


# DRIVERS


def _resolve_method(method: str | IntegrationMethod | None) -> IntegrationMethod:
    return IntegrationMethod.from_value(
        Config.INTEGRATION_METHOD if method is None else method
    )


def _batches(
    points: ndarray, weights: ndarray, batch_size: int
) -> Iterator[tuple[ndarray, ndarray]]:
    for start in range(0, len(weights), batch_size):
        stop = start + batch_size
        yield points[start:stop], weights[start:stop]


def _integrate_pointwise(
    batches: Iterator[tuple[ndarray, ndarray]],
    functions: list[float | Function],
    project: Callable,
    out: ndarray,
) -> ndarray:
    for points, weights in batches:
        coords = [points] if points.ndim == 1 else [points[:, 0], points[:, 1]]
        project(points, weights, _eval_load_functions(functions, coords))
    return out


def _zeros_1d(N: int, zero_order: bool) -> ndarray:
    return np.zeros((2,) if zero_order else (N, 2), dtype=float)


def _zeros_2d(shape: tuple, zero_order: bool) -> ndarray:
    M, N = shape
    return np.zeros(((M + N) if zero_order else (M * N), 3), dtype=float)


def integrate_line_1d(
    L: float,
    N: int,
    domain: ndarray,
    values: Iterable,
    *,
    zero_order: bool = False,
    method: str | IntegrationMethod | None = None,
    num_mc: int | None = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over a line segment of a beam.
    Values are expected in the order [f, m].
    """
    out = _zeros_1d(N, zero_order)
    if zero_order and _has_no_moment_load(values):
        return out
    functions = _load_functions(values, ["x"])

    def project(x, w, f):
        _project_1d(L, N, x, w, f, out, zero_order)

    a, b = domain
    if _resolve_method(method) is IntegrationMethod.MONTE_CARLO:
        sampler = partial(generate_random_points_on_line_segment_1d, a, b)
        batches = monte_carlo_samples(
            sampler,
            num_mc or Config.NUM_MC_SAMPLES_BEAM,
            Config.MC_BATCH_SIZE_BEAM,
            b - a,
            rng,
        )
    else:
        x, w = line_1d_rule(L, N, domain)
        batches = _batches(x, w, Config.QUADRATURE_BATCH_SIZE)
    return _integrate_pointwise(batches, functions, project, out)


def integrate_line_2d(
    size: tuple,
    shape: tuple,
    domain: ndarray,
    values: Iterable,
    *,
    zero_order: bool = False,
    method: str | IntegrationMethod | None = None,
    num_mc: int | None = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over a line segment of a plate.
    Values are expected in the order [f, mx, my].
    """
    out = _zeros_2d(shape, zero_order)
    if zero_order and _has_no_moment_load(values):
        return out
    functions = _load_functions(values, ["x", "y"])

    def project(xy, w, f):
        _project_2d(size, shape, xy, w, f, out, zero_order)

    if _resolve_method(method) is IntegrationMethod.MONTE_CARLO:
        sampler = partial(generate_random_points_on_line_segment_2d, *domain)
        batches = monte_carlo_samples(
            sampler,
            num_mc or Config.NUM_MC_SAMPLES_PLATE,
            Config.MC_BATCH_SIZE_PLATE,
            np.linalg.norm(domain[1] - domain[0]),
            rng,
        )
    else:
        points, w = line_2d_rule(size, shape, domain)
        batches = _batches(points, w, Config.QUADRATURE_BATCH_SIZE)
    return _integrate_pointwise(batches, functions, project, out)


def integrate_rectangle(
    size: tuple,
    shape: tuple,
    domain: ndarray,
    values: Iterable,
    *,
    zero_order: bool = False,
    method: str | IntegrationMethod | None = None,
    num_mc: int | None = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over a rectangle, given by two
    opposite corners. Values are expected in the order [f, mx, my].
    """
    out = _zeros_2d(shape, zero_order)
    if zero_order and _has_no_moment_load(values):
        return out
    functions = _load_functions(values, ["x", "y"])
    if _resolve_method(method) is IntegrationMethod.MONTE_CARLO:
        lower, upper = np.min(domain, axis=0), np.max(domain, axis=0)
        sampler = partial(generate_random_points_in_rectangle, lower, upper)
        batches = monte_carlo_samples(
            sampler,
            num_mc or Config.NUM_MC_SAMPLES_PLATE,
            Config.MC_BATCH_SIZE_PLATE,
            np.prod(upper - lower),
            rng,
        )

        def project(xy, w, f):
            _project_2d(size, shape, xy, w, f, out, zero_order)

        return _integrate_pointwise(batches, functions, project, out)

    x, wx, y, wy = rectangle_rule(size, shape, domain)
    # the grid is processed in strips of rows, to limit the memory footprint
    rows = max(1, Config.QUADRATURE_BATCH_SIZE // len(y))
    for start in range(0, len(x), rows):
        xs, wxs = x[start : start + rows], wx[start : start + rows]
        X, Y = np.meshgrid(xs, y, indexing="ij")
        G = _eval_load_functions(functions, [X, Y])
        G *= (wxs[:, None] * wy[None, :])[:, :, None]
        _project_2d_tensor(size, shape, xs, y, G, out, zero_order)
    return out


def integrate_disk(
    size: tuple,
    shape: tuple,
    domain: tuple[ndarray, float],
    values: Iterable,
    *,
    zero_order: bool = False,
    method: str | IntegrationMethod | None = None,
    num_mc: int | None = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Returns coefficients for arbitrary loads over a disk, given by its center
    and radius. Values are expected in the order [f, mx, my].
    """
    center, radius = domain
    out = _zeros_2d(shape, zero_order)
    if zero_order and _has_no_moment_load(values):
        return out
    functions = _load_functions(values, ["x", "y"])

    def project(xy, w, f):
        _project_2d(size, shape, xy, w, f, out, zero_order)

    if _resolve_method(method) is IntegrationMethod.MONTE_CARLO:
        sampler = partial(generate_random_points_in_disk, center, radius)
        batches = monte_carlo_samples(
            sampler,
            num_mc or Config.NUM_MC_SAMPLES_PLATE,
            Config.MC_BATCH_SIZE_PLATE,
            np.pi * radius**2,
            rng,
        )
    else:
        points, w = disk_rule(size, shape, center, radius)
        batches = _batches(points, w, Config.QUADRATURE_BATCH_SIZE)
    return _integrate_pointwise(batches, functions, project, out)
