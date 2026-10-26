from numbers import Integral, Real
from types import NoneType
from typing import Any, Iterable

import numpy as np
from numpy import ndarray

RNGType = int | np.random.Generator | NoneType

# relative tolerance of the checks that points lie within the domain of a problem
DOMAIN_TOLERANCE = 1e-8


def is_scalar(value: any) -> bool:
    """
    Check if a value is a scalar.
    """
    return isinstance(value, (int, float))


def to_positive_float(value: Any, name: str) -> float:
    """
    Returns a value as a float, or raises an error if it is not a positive,
    finite scalar.
    """
    if isinstance(value, ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{name}' must be a scalar, got {value!r}.")
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"'{name}' must be positive and finite, got {value}.")
    return value


def to_positive_int(value: Any, name: str) -> int:
    """
    Returns a value as an int, or raises an error if it is not a positive integer.
    """
    if isinstance(value, ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"'{name}' must be an integer, got {value!r}.")
    if value < 1:
        raise ValueError(f"'{name}' must be positive, got {value}.")
    return int(value)


def check_points_in_domain(
    points: Iterable, size: float | Iterable[float], what: str = "points"
) -> ndarray:
    """
    Raises a `ValueError` if any of the points is outside of the domain of a
    problem of the given size, otherwise returns the points as a 2d float array
    of shape (nP, nD), where nD is 1 for a beam and 2 for a plate. For a plate,
    further coordinates (like the z coordinates of a mesh) are dropped.
    """
    size = np.atleast_1d(np.asarray(size, dtype=float))
    dim = len(size)
    points = np.asarray(points, dtype=float)
    if dim == 1:
        points = points.reshape(-1, 1)
    else:
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] < dim:
            raise ValueError(
                f"The {what} must be given as an array of shape (n, {dim}), "
                f"got shape {points.shape}."
            )
        points = points[:, :dim]
    tol = DOMAIN_TOLERANCE * size
    inside = np.isfinite(points) & (points >= -tol) & (points <= size + tol)
    if not np.all(inside):
        bounds = " x ".join(f"[0, {L}]" for L in size)
        raise ValueError(f"The {what} must lie within the domain {bounds}.")
    return points


def points_to_rectangle_region(points: ndarray) -> tuple[float]:
    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    return xmin, ymin, xmax - xmin, ymax - ymin


def generate_random_points_in_disk(
    center: tuple[float, float],
    radius: float,
    N: int,
    *,
    out: ndarray | NoneType = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Generate N random points uniformly distributed within a disk.

    The random numbers are drawn from `rng`, see :func:`numpy.random.default_rng`
    for the accepted values. The global random state is left untouched.
    """
    rng = np.random.default_rng(rng)
    # Generate N random angles uniformly between 0 and 2*pi
    angles = rng.uniform(0, 2 * np.pi, N)

    # Generate N random radii uniformly distributed within the disk (use sqrt to ensure uniformity)
    radii = radius * np.sqrt(rng.uniform(0, 1, N))

    # Calculate x and y coordinates based on polar coordinates (r, theta)
    x_points = center[0] + radii * np.cos(angles)
    y_points = center[1] + radii * np.sin(angles)

    # Combine x and y into a single array of shape (N, 2)
    if out is None:
        out = np.column_stack((x_points, y_points))
    else:
        out[:, 0] = x_points
        out[:, 1] = y_points

    return out


def generate_random_points_in_rectangle(
    bottom_left: tuple[float, float],
    upper_right: tuple[float, float],
    N: int,
    *,
    out: ndarray | NoneType = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Generate N random points uniformly distributed within a rectangle.

    The random numbers are drawn from `rng`, see :func:`numpy.random.default_rng`
    for the accepted values. The global random state is left untouched.
    """
    rng = np.random.default_rng(rng)
    # Extract the x and y limits from the bottom-left and upper-right points
    x_min, y_min = bottom_left
    x_max, y_max = upper_right

    # Combine x and y into a single array of shape (N, 2)
    if out is None:
        x_points = rng.uniform(x_min, x_max, N)
        y_points = rng.uniform(y_min, y_max, N)
        out = np.column_stack((x_points, y_points))
    else:
        out[:, 0] = rng.uniform(x_min, x_max, N)
        out[:, 1] = rng.uniform(y_min, y_max, N)

    return out


def generate_random_points_on_line_segment_1d(
    start: float,
    end: float,
    N: int,
    *,
    out: ndarray | NoneType = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Generate N points uniformly distributed along a line segment in 1d.

    The random numbers are drawn from `rng`, see :func:`numpy.random.default_rng`
    for the accepted values. The global random state is left untouched.
    """
    rng = np.random.default_rng(rng)
    # Generate N random numbers between 0 and 1 to interpolate between start and end
    t = rng.uniform(0, 1, N)

    # Interpolate between start and end points using the parameter t
    # Combine x and y into a single array of shape (N, 2)
    if out is None:
        out = (1 - t) * start + t * end
    else:
        out[:] = (1 - t) * start + t * end

    return out


def generate_random_points_on_line_segment_2d(
    start: tuple[float, float],
    end: tuple[float, float],
    N: int,
    *,
    out: ndarray | NoneType = None,
    rng: RNGType = None,
) -> ndarray:
    """
    Generate N points uniformly distributed along a line segment in 2d.

    The random numbers are drawn from `rng`, see :func:`numpy.random.default_rng`
    for the accepted values. The global random state is left untouched.
    """
    rng = np.random.default_rng(rng)
    # Convert start and end points to numpy arrays
    start = np.array(start)
    end = np.array(end)

    # Generate N random numbers between 0 and 1 to interpolate between start and end
    t = rng.uniform(0, 1, N)

    # Interpolate between start and end points using the parameter t
    # Combine x and y into a single array of shape (N, 2)
    if out is None:
        out = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end
    else:
        out[:, :] = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end

    return out
