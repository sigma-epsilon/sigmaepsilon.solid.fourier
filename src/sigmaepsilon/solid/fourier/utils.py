from types import NoneType

import numpy as np
from numpy import ndarray


def is_scalar(value: any) -> bool:
    """
    Check if a value is a scalar.
    """
    return isinstance(value, (int, float))


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
    out: ndarray | NoneType = None
) -> ndarray:
    """
    Generate N random points uniformly distributed within a disk.
    """
    # Generate N random angles uniformly between 0 and 2*pi
    angles = np.random.uniform(0, 2 * np.pi, N)

    # Generate N random radii uniformly distributed within the disk (use sqrt to ensure uniformity)
    radii = radius * np.sqrt(np.random.uniform(0, 1, N))

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
    out: ndarray | NoneType = None
) -> ndarray:
    """
    Generate N random points uniformly distributed within a rectangle.
    """
    # Extract the x and y limits from the bottom-left and upper-right points
    x_min, y_min = bottom_left
    x_max, y_max = upper_right

    # Combine x and y into a single array of shape (N, 2)
    if out is None:
        x_points = np.random.uniform(x_min, x_max, N)
        y_points = np.random.uniform(y_min, y_max, N)
        out = np.column_stack((x_points, y_points))
    else:
        out[:, 0] = np.random.uniform(x_min, x_max, N)
        out[:, 1] = np.random.uniform(y_min, y_max, N)

    return out


def generate_random_points_on_line_segment_1d(
    start: float, end: float, N: int, *, out: ndarray | NoneType = None
) -> ndarray:
    """
    Generate N points uniformly distributed along a line segment in 1d.
    """
    # Generate N random numbers between 0 and 1 to interpolate between start and end
    t = np.random.uniform(0, 1, N)

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
    out: ndarray | NoneType = None
) -> ndarray:
    """
    Generate N points uniformly distributed along a line segment in 2d.
    """
    # Convert start and end points to numpy arrays
    start = np.array(start)
    end = np.array(end)

    # Generate N random numbers between 0 and 1 to interpolate between start and end
    t = np.random.uniform(0, 1, N)

    # Interpolate between start and end points using the parameter t
    # Combine x and y into a single array of shape (N, 2)
    if out is None:
        out = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end
    else:
        out[:, :] = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end

    return out
