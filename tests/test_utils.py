import numpy as np

from sigmaepsilon.solid.fourier.utils import (
    generate_random_points_in_disk,
    generate_random_points_in_rectangle,
    generate_random_points_on_line_segment_1d,
    generate_random_points_on_line_segment_2d,
)


def test_random_points_in_disk():
    center = (0.0, 0.0)
    radius = 1.0
    N = 1000
    points = generate_random_points_in_disk(center, radius, N)
    generate_random_points_in_disk(center, radius, N, out=points)
    assert points.shape == (N, 2)
    assert np.all(np.linalg.norm(points, axis=1) <= radius)


def test_random_points_in_rectangle():
    bottom_left = (0.0, 0.0)
    upper_right = (1.0, 1.0)
    N = 1000
    points = generate_random_points_in_rectangle(bottom_left, upper_right, N)
    generate_random_points_in_rectangle(bottom_left, upper_right, N, out=points)
    assert points.shape == (N, 2)
    assert np.all(np.logical_and(bottom_left <= points, points <= upper_right))


def test_random_points_line_1d():
    start = 0.0
    end = 1.0
    N = 1000
    points = generate_random_points_on_line_segment_1d(start, end, N)
    generate_random_points_on_line_segment_1d(start, end, N, out=points)
    assert points.shape == (N,)
    assert np.all(np.logical_and(start <= points, points <= end))


def test_random_points_line_2d():
    start = (0.0, 0.0)
    end = (1.0, 1.0)
    N = 1000
    points = generate_random_points_on_line_segment_2d(start, end, N)
    generate_random_points_on_line_segment_2d(start, end, N, out=points)
    assert points.shape == (N, 2)
    assert np.all(np.logical_and(start <= points, points <= end))


def test_random_points_use_the_given_rng():
    np.random.seed(0)
    state = np.random.get_state()[1].copy()
    a = generate_random_points_in_disk((0.0, 0.0), 1.0, 10, rng=1)
    b = generate_random_points_in_disk((0.0, 0.0), 1.0, 10, rng=1)
    c = generate_random_points_on_line_segment_1d(0.0, 1.0, 10, rng=1)
    d = generate_random_points_on_line_segment_1d(0.0, 1.0, 10, rng=1)
    assert np.array_equal(a, b)
    assert np.array_equal(c, d)
    assert np.array_equal(state, np.random.get_state()[1])
