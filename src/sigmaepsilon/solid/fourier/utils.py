import numpy as np


def points_to_rectangle_region(points: np.ndarray) -> tuple[float]:
    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    return xmin, ymin, xmax - xmin, ymax - ymin


def sin1d(x, i=1, L=1):
    return np.sin(x * np.pi * i / L)


def cos1d(x, i=1, L=1):
    return np.cos(x * np.pi * i / L)
