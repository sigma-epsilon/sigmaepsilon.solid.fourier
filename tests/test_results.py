"""Tests of reading the results of linear static analyses by location."""

import unittest

import numpy as np
from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from sigmaepsilon.solid.fourier import (
    LineLoad,
    LoadGroup,
    NavierBeam,
    NavierPlate,
    PointLoad,
    RectangleLoad,
)
from sigmaepsilon.solid.fourier.result import (
    BeamLoadCaseResultLinStat,
    PlateLoadCaseResultLinStat,
    _unique_with_tolerance,
)


def _plate_problem(shear: bool) -> tuple[NavierPlate, LoadGroup]:
    size = Lx, Ly = (600.0, 800.0)
    D = np.array([[1e6, 2e5, 0], [2e5, 1e6, 0], [0, 0, 4e5]])
    S = np.diag([1e5, 1e5]) if shear else None
    plate = NavierPlate(size, (20, 20), D=D, S=S)
    loads = LoadGroup(
        LG=LoadGroup(
            LC1=RectangleLoad([[0, 0], [Lx, Ly]], [-0.1, 0, 0]),
            LC2=PointLoad([Lx / 3, Ly / 2], [0, 10.0, 5.0]),
        )
    )
    return plate, loads


def _grid_points(Lx: float, Ly: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = np.linspace(0, Lx, 4), np.linspace(0, Ly, 5)
    points = np.array([[x, y] for y in ys for x in xs])
    return xs, ys, points


class TestPlateResults(SigmaEpsilonTestCase):
    """Results of plates."""

    def test_at_reproduces_and_extends_the_analysis(self):
        for shear in (False, True):
            plate, loads = _plate_problem(shear)
            _, _, points = _grid_points(*plate.size)
            solution = plate.linear_static_analysis(points, loads)
            point = [123.4, 567.8]
            reference = plate.linear_static_analysis([point], loads)
            for case in ("LC1", "LC2"):
                result = solution["LG", case]
                self.assertTrue(np.allclose(result.at(points).values, result.values))
                at_point = result.at(point)
                self.assertIsInstance(at_point, PlateLoadCaseResultLinStat)
                self.assertTrue(
                    np.allclose(at_point.values, reference["LG", case].values)
                )
                self.assertTrue(np.allclose(at_point.points, [point]))
                self.assertEqual(at_point.name, result.name)
                # the evaluator is passed on, so the new results are evaluable too
                self.assertTrue(np.allclose(at_point.at(point).values, at_point.values))

    def test_at_is_not_affected_by_later_changes_of_the_problem(self):
        plate, loads = _plate_problem(True)
        _, _, points = _grid_points(*plate.size)
        result = plate.linear_static_analysis(points, loads)["LG", "LC1"]
        plate.D = 2 * plate.D
        self.assertTrue(np.allclose(result.at(points).values, result.values))

    def test_at_out_of_domain(self):
        plate, loads = _plate_problem(False)
        _, _, points = _grid_points(*plate.size)
        result = plate.linear_static_analysis(points, loads)["LG", "LC1"]
        with self.assertRaises(ValueError):
            result.at([1000.0, 0.0])

    def test_at_without_evaluator(self):
        result = PlateLoadCaseResultLinStat(np.zeros((2, 13)))
        with self.assertRaises(ValueError):
            result.at([0.0, 0.0])

    def test_component_access(self):
        plate, loads = _plate_problem(False)
        _, _, points = _grid_points(*plate.size)
        result = plate.linear_static_analysis(points, loads)["LG", "LC1"]
        self.assertTrue(np.allclose(result["UZ"], result.values[:, 0]))
        self.assertTrue(np.allclose(result[["MX", "QY"]], result.values[:, [8, 12]]))
        with self.assertRaises(KeyError):
            _ = result["UY"]

    def test_points_and_coordinates(self):
        plate, loads = _plate_problem(False)
        _, _, points = _grid_points(*plate.size)
        result = plate.linear_static_analysis(points, loads)["LG", "LC1"]
        self.assertTrue(np.allclose(result.points, points))

        xarr = result.to_xarray()
        self.assertEqual(xarr.dims, ("index", "component"))
        self.assertTrue(np.allclose(xarr.coords["x"].values, points[:, 0]))
        self.assertTrue(np.allclose(xarr.coords["y"].values, points[:, 1]))
        self.assertTrue(np.isclose(xarr.loc[9, "UZ"].item(), result["UZ"][9]))

        df = result.to_pandas()
        self.assertEqual(list(df.columns), list(result.components))
        df = result.to_pandas(coords=True)
        self.assertEqual(list(df.columns), ["x", "y", *result.components])
        self.assertTrue(np.allclose(df["y"].values, points[:, 1]))

    def test_unknown_points(self):
        result = PlateLoadCaseResultLinStat(np.zeros((2, 13)))
        self.assertIsNone(result.points)
        self.assertNotIn("x", result.to_xarray().coords)
        self.assertEqual(list(result.to_pandas(coords=True).columns), result.components)
        with self.assertRaises(ValueError):
            result.to_xarray(grid=True)

    def test_to_xarray_grid(self):
        plate, loads = _plate_problem(True)
        xs, ys, points = _grid_points(*plate.size)
        # the order of the points must not matter
        points = points[np.random.default_rng(0).permutation(len(points))]
        result = plate.linear_static_analysis(points, loads)["LG", "LC2"]
        xarr = result.to_xarray(grid=True)
        self.assertEqual(xarr.dims, ("x", "y", "component"))
        self.assertTrue(np.allclose(xarr.coords["x"].values, xs))
        self.assertTrue(np.allclose(xarr.coords["y"].values, ys))
        for i, (x, y) in enumerate(points):
            value = xarr.sel(x=x, y=y, method="nearest").sel(component="MX").item()
            self.assertTrue(np.isclose(value, result["MX"][i]))

    def test_to_xarray_grid_invalid(self):
        plate, loads = _plate_problem(False)
        _, _, points = _grid_points(*plate.size)
        # a missing point
        result = plate.linear_static_analysis(points[:-1], loads)["LG", "LC1"]
        with self.assertRaises(ValueError):
            result.to_xarray(grid=True)
        # a duplicate point
        points = np.vstack([points[:-1], points[:1]])
        result = plate.linear_static_analysis(points, loads)["LG", "LC1"]
        with self.assertRaises(ValueError):
            result.to_xarray(grid=True)


class TestBeamResults(SigmaEpsilonTestCase):
    """Results of beams."""

    def _solve(self, x):
        beam = NavierBeam(10.0, 50, EI=1e3, GA=1e4)
        loads = LoadGroup(LC=LineLoad([0, 10.0], [1.0, 0.0]))
        return beam, loads, beam.linear_static_analysis(x, loads)["LC"]

    def test_at(self):
        x = np.linspace(0, 10, 11)
        beam, loads, result = self._solve(x)
        self.assertTrue(np.allclose(result.at(x).values, result.values))
        at_point = result.at(4.6)
        self.assertIsInstance(at_point, BeamLoadCaseResultLinStat)
        reference = beam.linear_static_analysis(4.6, loads)["LC"]
        self.assertTrue(np.allclose(at_point.values, reference.values))
        self.assertTrue(np.allclose(at_point.points, [[4.6]]))
        with self.assertRaises(ValueError):
            result.at(11.0)
        with self.assertRaises(ValueError):
            result.at([[1.0, 2.0]])

    def test_to_xarray_indexed_by_x(self):
        x = np.linspace(0, 10, 11)
        _, _, result = self._solve(x)
        xarr = result.to_xarray()
        value = xarr.sel(x=4.6, method="nearest").sel(component="UY").item()
        self.assertTrue(np.isclose(value, result["UY"][5]))
        self.assertTrue(np.isclose(xarr.loc[9, "UY"].item(), result["UY"][9]))
        df = result.to_pandas(coords=True)
        self.assertEqual(list(df.columns), ["x", *result.components])

    def test_to_xarray_not_monotonic(self):
        x = np.array([5.0, 1.0, 8.0, 1.0])
        _, _, result = self._solve(x)
        xarr = result.to_xarray()
        self.assertTrue(np.allclose(xarr.coords["x"].values, x))
        self.assertNotIn("x", xarr.indexes)


class TestUtils(unittest.TestCase):
    """Helpers of the results."""

    def test_unique_with_tolerance(self):
        values = np.array([2.0, 1.0, 2.0 + 1e-12, 0.0])
        unique, labels = _unique_with_tolerance(values)
        self.assertTrue(np.allclose(unique, [0.0, 1.0, 2.0]))
        self.assertEqual(labels.tolist(), [2, 1, 2, 0])
        unique, labels = _unique_with_tolerance(np.zeros(0))
        self.assertEqual(len(unique), 0)
        self.assertEqual(len(labels), 0)


if __name__ == "__main__":
    unittest.main()
