"""
Deterministic quadrature rules for the integration domains of loads.

The number of nodes is derived from the highest harmonic involved, since the
integrands oscillate with a frequency up to :math:`\\pi M / L_x` and
:math:`\\pi N / L_y`. Composite Gauss-Legendre rules with
`Config.QUADRATURE_PANELS_PER_HALF_WAVE` panels per half-wavelength and
`Config.QUADRATURE_ORDER` nodes per panel are used along straight directions,
and the trapezoidal rule (which is spectrally accurate for periodic integrands)
in the circumferential direction of disks.
"""

from functools import lru_cache

import numpy as np
from numpy import ndarray

from .config import Config

__all__ = [
    "disk_rule",
    "gauss_legendre",
    "line_1d_rule",
    "line_2d_rule",
    "rectangle_rule",
]


@lru_cache(maxsize=32)
def _leggauss(order: int) -> tuple[ndarray, ndarray]:
    return np.polynomial.legendre.leggauss(order)


def _num_panels(half_waves: float) -> int:
    return max(1, int(np.ceil(Config.QUADRATURE_PANELS_PER_HALF_WAVE * half_waves)))


def gauss_legendre(
    a: float, b: float, num_panels: int = 1, order: int | None = None
) -> tuple[ndarray, ndarray]:
    """
    Returns the nodes and weights of a composite Gauss-Legendre rule over the
    interval [a, b], with `num_panels` panels of equal length and `order` nodes
    per panel.
    """
    order = order or Config.QUADRATURE_ORDER
    xi, wi = _leggauss(order)
    edges = np.linspace(a, b, num_panels + 1)
    half = (edges[1:] - edges[:-1]) / 2
    mid = (edges[1:] + edges[:-1]) / 2
    nodes = (mid[:, None] + half[:, None] * xi[None, :]).ravel()
    weights = (half[:, None] * wi[None, :]).ravel()
    return nodes, weights


def line_1d_rule(L: float, N: int, domain: ndarray) -> tuple[ndarray, ndarray]:
    """
    Returns the nodes and weights for a line segment of a beam of length `L`,
    with `N` harmonics.
    """
    a, b = domain
    return gauss_legendre(a, b, _num_panels(N * abs(b - a) / L))


def line_2d_rule(size: tuple, shape: tuple, domain: ndarray) -> tuple[ndarray, ndarray]:
    """
    Returns the nodes as a 2d array of shape (nP, 2) and the weights for
    a line segment on a plate.
    """
    Lx, Ly = size
    M, N = shape
    start, end = np.asarray(domain[0], dtype=float), np.asarray(domain[1], dtype=float)
    dx, dy = np.abs(end - start)
    half_waves = M * dx / Lx + N * dy / Ly
    t, w = gauss_legendre(0.0, 1.0, _num_panels(half_waves))
    nodes = start[None, :] + t[:, None] * (end - start)[None, :]
    return nodes, w * np.hypot(dx, dy)


def rectangle_rule(
    size: tuple, shape: tuple, domain: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Returns a tensor-product rule over a rectangle as the nodes and weights
    in the x direction, followed by the nodes and weights in the y direction.
    """
    Lx, Ly = size
    M, N = shape
    (x0, y0), (x1, y1) = np.min(domain, axis=0), np.max(domain, axis=0)
    x, wx = gauss_legendre(x0, x1, _num_panels(M * (x1 - x0) / Lx))
    y, wy = gauss_legendre(y0, y1, _num_panels(N * (y1 - y0) / Ly))
    return x, wx, y, wy


def disk_rule(
    size: tuple, shape: tuple, center: ndarray, radius: float
) -> tuple[ndarray, ndarray]:
    """
    Returns the nodes as a 2d array of shape (nP, 2) and the weights for
    a disk, using a Gauss-Legendre rule in the radial direction and the
    trapezoidal rule in the circumferential direction.
    """
    Lx, Ly = size
    M, N = shape
    order = Config.QUADRATURE_ORDER
    half_waves = radius * (M / Lx + N / Ly)
    r, wr = gauss_legendre(0.0, radius, _num_panels(half_waves))
    # along the circumference the integrand has an angular bandwidth of up
    # to pi * half_waves, the trapezoidal rule converges exponentially once
    # the number of nodes exceeds it
    num_theta = max(4 * order, 2 * _num_panels(np.pi * half_waves) + 2 * order)
    theta = np.arange(num_theta) * (2 * np.pi / num_theta)
    wt = 2 * np.pi / num_theta
    x = center[0] + np.outer(r, np.cos(theta))
    y = center[1] + np.outer(r, np.sin(theta))
    w = np.outer(wr * r, np.full(num_theta, wt))
    return np.column_stack((x.ravel(), y.ravel())), w.ravel()
