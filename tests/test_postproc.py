import unittest
from unittest import mock

import numpy as np
from sigmaepsilon.core.testing import SigmaEpsilonTestCase

import sigmaepsilon.solid.fourier.postproc_gemm as postproc_gemm
from sigmaepsilon.solid.fourier import (
    LoadGroup,
    NavierPlate,
    PointLoad,
    RectangleLoad,
)
from sigmaepsilon.solid.fourier.config import Config
from sigmaepsilon.solid.fourier.enums import PostprocMethod
from sigmaepsilon.solid.fourier.postproc import (
    postproc,
    postproc_Kirchhoff,
    postproc_Mindlin,
)
from sigmaepsilon.solid.fourier.postproc_gemm import (
    postproc_Kirchhoff_gemm,
    postproc_Mindlin_gemm,
)


def max_relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    """
    Returns the largest error over all components, each component being scaled
    by the largest absolute value of the reference.
    """
    scale = np.max(np.abs(reference), axis=(0, 1, 2))
    scale[scale == 0] = 1.0
    return np.max(np.max(np.abs(values - reference), axis=(0, 1, 2)) / scale)


class TestGemmVsNumba(SigmaEpsilonTestCase):
    """
    The matrix product formulation is compared with the Numba kernels, with
    random coefficients and several left- and right-hand sides.
    """

    size = np.array([3.0, 5.0])
    # different number of harmonics in the two directions, so that
    # mixing up the harmonic indices does not go unnoticed
    shape = np.array([7, 9])
    nLHS, nRHS = 2, 3

    def setUp(self):
        rng = np.random.default_rng(0)
        M, N = self.shape
        nLHS, nRHS = self.nLHS, self.nRHS
        self.D = np.zeros((nLHS, 3, 3))
        self.S = np.zeros((nLHS, 2, 2))
        for i in range(nLHS):
            self.D[i] = [[2.0 + i, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.2 + i]]
            self.S[i] = [[4.0, 0.0], [0.0, 6.0 + i]]
        self.solution_M = rng.normal(size=(nLHS, nRHS, M * N, 3))
        self.solution_K = rng.normal(size=(nLHS, nRHS, M * N))
        self.loads = rng.normal(size=(nRHS, M * N, 3))
        self.scattered = rng.uniform(size=(50, 2)) * self.size
        x, y = np.meshgrid(np.linspace(0, 3, 11), np.linspace(0, 5, 13), indexing="ij")
        # the points of a grid, in a shuffled order
        self.grid = rng.permutation(np.column_stack([x.ravel(), y.ravel()]))
        # a partial grid with some duplicate points
        partial = self.grid[:100]
        self.partial = np.concatenate([partial, partial[:5]])

    def assert_match(self, points: np.ndarray, grid: bool | None):
        args = (self.size, self.shape, points)
        reference = postproc_Mindlin(*args, self.solution_M, self.D, self.S)
        values = postproc_Mindlin_gemm(
            *args, self.solution_M, self.D, self.S, grid=grid
        )
        self.assertEqual(values.shape, reference.shape)
        self.assertLess(max_relative_error(values, reference), 1e-12)
        reference = postproc_Kirchhoff(*args, self.solution_K, self.D, self.loads)
        values = postproc_Kirchhoff_gemm(
            *args, self.solution_K, self.D, self.loads, grid=grid
        )
        self.assertEqual(values.shape, reference.shape)
        self.assertLess(max_relative_error(values, reference), 1e-12)

    def test_scattered_points(self):
        for grid in (None, True, False):
            self.assert_match(self.scattered, grid)

    def test_grid(self):
        for grid in (None, True, False):
            self.assert_match(self.grid, grid)

    def test_partial_grid_with_duplicates(self):
        for grid in (None, True, False):
            self.assert_match(self.partial, grid)

    def test_small_buffer(self):
        # the points are processed in several batches and the grid in several strips
        with mock.patch.object(Config, "POSTPROC_BUFFER_SIZE", 500):
            for grid in (True, False):
                self.assert_match(self.grid, grid)

    def test_no_points(self):
        points = np.zeros((0, 2))
        values = postproc_Mindlin_gemm(
            self.size, self.shape, points, self.solution_M, self.D, self.S
        )
        self.assertEqual(values.shape, (self.nLHS, self.nRHS, 0, 13))


class TestGridDetection(SigmaEpsilonTestCase):
    def test_grid_is_chosen_for_grids(self):
        self.assertTrue(postproc_gemm._use_grid(300 * 400, 300, 400, 60, 60))

    def test_grid_is_not_chosen_for_scattered_points(self):
        self.assertFalse(postproc_gemm._use_grid(10000, 10000, 10000, 60, 60))

    def test_automatic_choice(self):
        test = TestGemmVsNumba()
        test.setUp()
        args = (test.size, test.shape)
        for points, expected in [(test.grid, True), (test.scattered, False)]:
            with mock.patch.object(
                postproc_gemm, "_evaluate_grid", wraps=postproc_gemm._evaluate_grid
            ) as evaluate_grid:
                postproc_Mindlin_gemm(*args, points, test.solution_M, test.D, test.S)
                self.assertEqual(evaluate_grid.called, expected)


class TestPostprocMethod(SigmaEpsilonTestCase):
    size = (2.0, 3.0)
    shape = (10, 14)
    D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])

    def loads(self) -> LoadGroup:
        domain = [[0.5, 0.7], [1.3, 1.9]]
        point = [0.8, 1.1]
        return LoadGroup(
            fz=RectangleLoad(domain, [1.0, 0, 0]),
            mx=RectangleLoad(domain, [0, 1.0, 0]),
            my=RectangleLoad(domain, [0, 0, 1.0]),
            Mx=PointLoad(point, [0, 1.0, 0]),
            My=PointLoad(point, [0, 0, 1.0]),
        )

    def test_default_is_blas(self):
        self.assertIs(
            PostprocMethod.from_value(Config.POSTPROC_METHOD), PostprocMethod.BLAS
        )

    def test_methods_agree_end_to_end(self):
        x, y = np.meshgrid(np.linspace(0, 2, 9), np.linspace(0, 3, 7))
        points = np.column_stack([x.ravel(), y.ravel()])
        loads = self.loads()
        for S in (None, 1e3 * np.eye(2)):
            plate = NavierPlate(self.size, self.shape, D=self.D, S=S)
            results = {}
            for method in ("blas", "numba"):
                with mock.patch.object(Config, "POSTPROC_METHOD", method):
                    results[method] = plate.linear_static_analysis(points, loads)
            for key in loads:
                error = max_relative_error(
                    results["blas"][key].values[None, None],
                    results["numba"][key].values[None, None],
                )
                self.assertLess(error, 1e-12, msg=f"load case {key}")

    def test_config_method(self):
        test = TestGemmVsNumba()
        test.setUp()
        args = (test.size, test.shape, test.scattered, test.solution_M, None, test.D)
        with mock.patch.object(Config, "POSTPROC_METHOD", PostprocMethod.BLAS):
            blas = postproc(*args, test.S)
        with mock.patch.object(Config, "POSTPROC_METHOD", "NUMBA"):
            numba = postproc(*args, test.S)
        self.assertLess(max_relative_error(blas, numba), 1e-12)

    def test_invalid_method(self):
        for method in ("cuda", 1):
            with self.assertRaises(ValueError):
                PostprocMethod.from_value(method)


if __name__ == "__main__":
    unittest.main()
