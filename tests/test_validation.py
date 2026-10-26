import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.fourier import (
    NavierBeam,
    NavierPlate,
    LoadGroup,
    PointLoad,
    LineLoad,
    RectangleLoad,
    DiskLoad,
)


class TestBeamValidation(SigmaEpsilonTestCase):
    L = 10.0

    def test_invalid_length(self):
        for length in (0.0, -1.0, np.inf):
            with self.assertRaises(ValueError):
                NavierBeam(length, 10, EI=1.0)
        with self.assertRaises(TypeError):
            NavierBeam("10", 10, EI=1.0)

    def test_invalid_number_of_harmonics(self):
        with self.assertRaises(ValueError):
            NavierBeam(self.L, 0, EI=1.0)
        with self.assertRaises(TypeError):
            NavierBeam(self.L, 2.5, EI=1.0)
        beam = NavierBeam(self.L, np.int64(10), EI=1.0)
        self.assertEqual(beam.N, 10)
        beam.N = np.array(12)
        self.assertEqual(beam.N, 12)
        with self.assertRaises(ValueError):
            beam.shape = -1

    def test_invalid_stiffness(self):
        with self.assertRaises(TypeError):
            NavierBeam(self.L, 10, EI=[1000.0, 2000.0])
        with self.assertRaises(ValueError):
            NavierBeam(self.L, 10, EI=-1.0)
        with self.assertRaises(ValueError):
            NavierBeam(self.L, 10, EI=1.0, GA=0.0)
        beam = NavierBeam(self.L, 10, EI=np.float64(2.0), GA=np.array(3.0))
        self.assertEqual((beam.EI, beam.GA), (2.0, 3.0))

    def test_points_are_required(self):
        beam = NavierBeam(self.L, 10, EI=1.0)
        loads = LoadGroup(F=PointLoad(self.L / 2, [1.0, 0.0]))
        with self.assertRaises(ValueError):
            beam.linear_static_analysis(loads)

    def test_points_outside(self):
        beam = NavierBeam(self.L, 10, EI=1.0)
        loads = LoadGroup(F=PointLoad(self.L / 2, [1.0, 0.0]))
        with self.assertRaises(ValueError):
            beam.linear_static_analysis([0.0, 2 * self.L], loads)
        with self.assertRaises(ValueError):
            beam.linear_static_analysis([[1.0, 2.0]], loads)
        beam.linear_static_analysis([0.0, self.L], loads)

    def test_loads_outside(self):
        beam = NavierBeam(self.L, 10, EI=1.0)
        for load in (
            PointLoad(2 * self.L, [1.0, 0.0]),
            PointLoad(-1.0, [1.0, 0.0]),
            LineLoad([2.0, 12.0], [1.0, 0.0]),
            LineLoad([2.0, 12.0], ["x", 0.0]),
        ):
            with self.assertRaises(ValueError):
                load.rhs(beam)

    def test_reversed_line_load(self):
        beam = NavierBeam(self.L, 10, EI=1.0)
        with self.assertRaises(ValueError):
            LineLoad([7.0, 2.0], [1.0, 0.0]).rhs(beam)


class TestPlateValidation(SigmaEpsilonTestCase):
    size = Lx, Ly = (2.0, 3.0)
    shape = (8, 8)

    def test_invalid_size_and_shape(self):
        D = np.eye(3)
        for size in ((2.0,), (2.0, -3.0), 2.0):
            with self.assertRaises((ValueError, TypeError)):
                NavierPlate(size, self.shape, D=D)
        for shape in ((8,), (8, 0), (8, 2.5), 8):
            with self.assertRaises((ValueError, TypeError)):
                NavierPlate(self.size, shape, D=D)

    def test_bending_coupling_terms(self):
        D = np.eye(3)
        D[0, 2] = D[2, 0] = 0.1
        with self.assertRaises(ValueError):
            NavierPlate(self.size, self.shape, D=D)
        D = np.eye(3)
        D[1, 2] = D[2, 1] = 0.1
        with self.assertRaises(ValueError):
            NavierPlate(self.size, self.shape, D=D)

    def test_invalid_bending_stiffness(self):
        D = np.eye(3)
        D[0, 1] = 0.3
        for value in (np.eye(2), D, -np.eye(3), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                NavierPlate(self.size, self.shape, D=value)
        # numerical noise is tolerated
        D = np.eye(3) * 1e6
        D[0, 2] = D[2, 0] = 1e-10
        NavierPlate(self.size, self.shape, D=D)

    def test_invalid_shear_stiffness(self):
        D = np.eye(3)
        for S in ([[1.0, 0.2], [0.2, 1.0]], np.eye(3), [[1.0, 0.0], [0.0, -1.0]]):
            with self.assertRaises(ValueError):
                NavierPlate(self.size, self.shape, D=D, S=S)
        plate = NavierPlate(self.size, self.shape, D=D, S=np.eye(2))
        with self.assertRaises(ValueError):
            plate.S = [[1.0, 0.5], [0.5, 1.0]]
        plate.S = None
        self.assertIsNone(plate.S)

    def test_points_are_required(self):
        plate = NavierPlate(self.size, self.shape, D=np.eye(3))
        loads = LoadGroup(F=PointLoad([1.0, 1.0], [1.0, 0, 0]))
        with self.assertRaises(ValueError):
            plate.linear_static_analysis(loads)

    def test_points(self):
        plate = NavierPlate(self.size, self.shape, D=np.eye(3))
        loads = LoadGroup(F=PointLoad([1.0, 1.0], [1.0, 0, 0]))
        with self.assertRaises(ValueError):
            plate.linear_static_analysis([[1.0, 4.0]], loads)
        with self.assertRaises(ValueError):
            plate.linear_static_analysis([[1.0]], loads)
        # a single point, and the z coordinates of a mesh are accepted
        single = plate.linear_static_analysis([1.0, 1.5], loads)["F"].values
        mesh = plate.linear_static_analysis([[1.0, 1.5, 0.0]], loads)["F"].values
        self.assertTrue(np.array_equal(single, mesh))

    def test_loads_outside(self):
        plate = NavierPlate(self.size, self.shape, D=np.eye(3))
        for load in (
            PointLoad([1.0, 3.5], [1.0, 0, 0]),
            LineLoad([[0.5, 0.5], [2.5, 0.5]], [1.0, 0, 0]),
            RectangleLoad([[0.5, 0.5], [2.5, 1.0]], [1.0, 0, 0]),
            RectangleLoad([[0.5, 0.5], [2.5, 1.0]], ["x", 0, 0]),
            DiskLoad(((1.0, 1.5), 1.2), [1.0, 0, 0]),
            DiskLoad(((1.0, 1.5), 0.0), [1.0, 0, 0]),
        ):
            with self.assertRaises(ValueError):
                load.rhs(plate)

    def test_loads_on_the_boundary(self):
        plate = NavierPlate(self.size, self.shape, D=np.eye(3))
        RectangleLoad([[0.0, 0.0], [self.Lx, self.Ly]], ["x", 0, 0]).rhs(plate)
        DiskLoad(((1.0, 1.5), 1.0), [1.0, 0, 0]).rhs(plate)
        PointLoad([self.Lx, self.Ly], [1.0, 0, 0]).rhs(plate)


if __name__ == "__main__":
    unittest.main()
