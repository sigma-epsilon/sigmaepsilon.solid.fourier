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

# Indices of the shear strains, which are identically zero for the
# Euler-Bernoulli and Kirchhoff-Love models.
BEAM_SHEAR_STRAINS = [3]
PLATE_SHEAR_STRAINS = [6, 7]


def max_relative_error(values: np.ndarray, reference: np.ndarray, skip=()) -> float:
    """
    Returns the largest error over all components, each component being scaled
    by the largest absolute value of the reference.
    """
    keep = [i for i in range(values.shape[-1]) if i not in skip]
    values, reference = values[..., keep], reference[..., keep]
    scale = np.max(np.abs(reference), axis=0)
    scale[scale < 1e-12] = 1.0
    return np.max(np.max(np.abs(values - reference), axis=0) / scale)


class TestBeamAccuracy(SigmaEpsilonTestCase):
    L = 10.0
    EI = 1.0

    def test_point_force_closed_form(self):
        L, EI = self.L, self.EI
        loads = LoadGroup(F=PointLoad(L / 2, [1.0, 0.0]))
        res = NavierBeam(L, 400, EI=EI).linear_static_analysis([L / 2], loads)
        self.assertAlmostEqual(res["F"].values[0, 0], L**3 / (48 * EI), delta=1e-6)

    def test_point_couple_closed_form(self):
        # a couple M0 at L/3 is reacted by a constant shear force of -M0/L,
        # and the bending moment is M0 * x / L on the left side of the couple
        L, M0 = self.L, 1.0
        loads = LoadGroup(M=PointLoad(L / 3, [0.0, M0]))
        x = np.array([L / 6, L / 2, 5 * L / 6])
        for GA in (None, 1e3):
            beam = NavierBeam(L, 2000, EI=self.EI, GA=GA)
            values = beam.linear_static_analysis(x, loads)["M"].values
            self.assertTrue(np.allclose(values[:, 5], -M0 / L))
            self.assertAlmostEqual(values[0, 4], M0 / 6, delta=2e-3)

    def test_timoshenko_constant_rotation_from_mean_moment(self):
        # a uniform moment load over the whole span causes no deflection, a
        # constant shear force -m and a constant rotation m / GA
        L, m, GA = self.L, 2.0, 50.0
        loads = LoadGroup(m=LineLoad([0, L], [0.0, m]))
        x = np.linspace(1.0, L - 1.0, 5)
        values = (
            NavierBeam(L, 100, EI=self.EI, GA=GA)
            .linear_static_analysis(x, loads)["m"]
            .values
        )
        self.assertTrue(np.allclose(values[:, 0], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(values[:, 1], m / GA))
        self.assertTrue(np.allclose(values[:, 5], -m))

    def test_bernoulli_vs_timoshenko_slender_limit(self):
        L = self.L
        loads = LoadGroup(
            F=PointLoad(L / 3, [1.0, 0.0]),
            M=PointLoad(L / 3, [0.0, 1.0]),
            q=LineLoad([2.0, 7.0], [1.0, 0.0]),
            m=LineLoad([2.0, 7.0], [0.0, 1.0]),
        )
        x = np.linspace(0.3, L - 0.3, 41)
        bernoulli = NavierBeam(L, 100, EI=self.EI)
        timoshenko = NavierBeam(L, 100, EI=self.EI, GA=1e8)
        res_B = bernoulli.linear_static_analysis(x, loads)
        res_T = timoshenko.linear_static_analysis(x, loads)
        for key in loads:
            error = max_relative_error(
                res_B[key].values, res_T[key].values, skip=BEAM_SHEAR_STRAINS
            )
            self.assertLess(error, 1e-6, msg=f"load case {key}")


class TestPlateAccuracy(SigmaEpsilonTestCase):
    size = Lx, Ly = (2.0, 3.0)
    D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])

    def points(self, n: int = 50) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.column_stack(
            [
                rng.uniform(0.1, self.Lx - 0.1, n),
                rng.uniform(0.1, self.Ly - 0.1, n),
            ]
        )

    def test_kirchhoff_square_plate_closed_form(self):
        # Timoshenko & Woinowsky-Krieger, uniform load, nu = 0.3
        nu = 0.3
        D = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2]])
        loads = LoadGroup(q=RectangleLoad([[0, 0], [1, 1]], [1.0, 0, 0]))
        plate = NavierPlate((1.0, 1.0), (40, 40), D=D)
        values = plate.linear_static_analysis([[0.5, 0.5]], loads)["q"].values
        self.assertAlmostEqual(values[0, 0], 0.00406, delta=1e-5)
        self.assertAlmostEqual(values[0, 8], 0.0479, delta=1e-4)

    def test_kirchhoff_vs_mindlin_thin_limit(self):
        # different number of harmonics in the two directions, so that
        # mixing up the harmonic indices does not go unnoticed
        shape = (10, 14)
        domain = [[0.5, 0.7], [1.3, 1.9]]
        point = [0.8, 1.1]
        loads = LoadGroup(
            fz=RectangleLoad(domain, [1.0, 0, 0]),
            mx=RectangleLoad(domain, [0, 1.0, 0]),
            my=RectangleLoad(domain, [0, 0, 1.0]),
            Fz=PointLoad(point, [1.0, 0, 0]),
            Mx=PointLoad(point, [0, 1.0, 0]),
            My=PointLoad(point, [0, 0, 1.0]),
        )
        points = self.points()
        kirchhoff = NavierPlate(self.size, shape, D=self.D)
        mindlin = NavierPlate(self.size, shape, D=self.D, S=1e9 * np.eye(2))
        res_K = kirchhoff.linear_static_analysis(points, loads)
        res_M = mindlin.linear_static_analysis(points, loads)
        for key in loads:
            error = max_relative_error(
                res_K[key].values, res_M[key].values, skip=PLATE_SHEAR_STRAINS
            )
            self.assertLess(error, 1e-5, msg=f"load case {key}")

    def test_point_moment_vs_small_rectangle(self):
        h = 1e-3
        x, y = 0.8, 1.1
        small = [[x - h / 2, y - h / 2], [x + h / 2, y + h / 2]]
        loads = LoadGroup(
            point_mx=PointLoad([x, y], [0, 1.0, 0]),
            rect_mx=RectangleLoad(small, [0, 1 / h**2, 0]),
            point_my=PointLoad([x, y], [0, 0, 1.0]),
            rect_my=RectangleLoad(small, [0, 0, 1 / h**2]),
        )
        for S in (None, 1e3 * np.eye(2)):
            plate = NavierPlate(self.size, (10, 14), D=self.D, S=S)
            res = plate.linear_static_analysis(self.points(), loads)
            for dof in ("mx", "my"):
                error = max_relative_error(
                    res[f"point_{dof}"].values, res[f"rect_{dof}"].values
                )
                self.assertLess(error, 1e-4, msg=f"{dof}, S={S}")

    def test_uniform_moment_load_closed_form(self):
        # a uniform moment load over the whole plate causes no deflection, and is
        # balanced by a constant shear force, apart from the edge effects
        full = [[0, 0], [self.Lx, self.Ly]]
        loads = LoadGroup(
            mx=RectangleLoad(full, [0, 1.0, 0]),
            my=RectangleLoad(full, [0, 0, 1.0]),
        )
        points = [[1.0, 1.5], [0.7, 0.4], [1.4, 2.5]]
        for S in (None, 1e4 * np.eye(2)):
            plate = NavierPlate(self.size, (200, 200), D=self.D, S=S)
            res = plate.linear_static_analysis(points, loads)
            mx, my = res["mx"].values, res["my"].values
            self.assertTrue(np.allclose(mx[:, 0], 0.0, atol=1e-12))
            self.assertTrue(np.allclose(my[:, 0], 0.0, atol=1e-12))
            self.assertTrue(np.allclose(mx[:, 12], -1.0, atol=1e-2))
            self.assertTrue(np.allclose(my[:, 11], 1.0, atol=1e-2))


class TestZeroOrderCoefficients(SigmaEpsilonTestCase):
    """
    Compares the Monte Carlo zero-order coefficients of moment loads with
    closed-form ones of equivalent loads.
    """

    plate = NavierPlate((2.0, 3.0), (10, 14), D=np.eye(3))

    def setUp(self):
        np.random.seed(0)

    def assertCoefficientsClose(self, values, reference, rtol):
        scale = np.max(np.abs(reference))
        self.assertGreater(scale, 0.0)
        self.assertLess(np.max(np.abs(values - reference)) / scale, rtol)

    def test_rectangle_load(self):
        domain = [[0.5, 0.7], [1.3, 1.9]]
        mc = RectangleLoad(domain, ["0", 1.0, -2.0], num_mc=200_000)
        const = RectangleLoad(domain, [0, 1.0, -2.0])
        self.assertCoefficientsClose(
            mc.rhs0(self.plate), const.rhs0(self.plate), rtol=1e-2
        )

    def test_line_load_on_plate(self):
        # a line load along y is the limit of a thin rectangle
        h = 1e-4
        line = LineLoad([[0.8, 0.5], [0.8, 2.0]], [0, 1.0, 1.0], num_mc=200_000)
        rect = RectangleLoad(
            [[0.8 - h / 2, 0.5], [0.8 + h / 2, 2.0]], [0, 1 / h, 1 / h]
        )
        self.assertCoefficientsClose(
            line.rhs0(self.plate), rect.rhs0(self.plate), rtol=1e-2
        )

    def test_line_load_on_beam(self):
        beam = NavierBeam(10.0, 10, EI=1.0)
        mc = LineLoad([2.0, 7.0], ["0", 1.5], num_mc=200_000)
        const = LineLoad([2.0, 7.0], [0.0, 1.5])
        self.assertTrue(np.allclose(mc.rhs0(beam), const.rhs0(beam)))
        self.assertTrue(np.allclose(const.rhs0(beam), [0.0, 0.75]))

    def test_disk_load(self):
        # a small disk is close to a point load with the same resultant
        r = 0.005
        area = np.pi * r**2
        disk = DiskLoad(((1.0, 1.5), r), [0, 1.0, 1.0], num_mc=200_000)
        point = PointLoad([1.0, 1.5], [0, area, area])
        self.assertCoefficientsClose(
            disk.rhs0(self.plate), point.rhs0(self.plate), rtol=1e-2
        )

    def test_no_moment_load(self):
        loads = [
            DiskLoad(((1.0, 1.5), 0.3), [1.0, 0, 0]),
            LineLoad([[0.8, 0.5], [0.8, 2.0]], [1.0, 0, 0]),
            RectangleLoad([[0.5, 0.7], [1.3, 1.9]], ["x", 0, 0]),
        ]
        for load in loads:
            self.assertTrue(np.all(load.rhs0(self.plate) == 0))


class TestLoadApproximation(SigmaEpsilonTestCase):
    """
    The Fourier series approximation of moment loads includes their
    zero-order terms.
    """

    def test_beam_moment_load(self):
        beam = NavierBeam(10.0, 200, EI=1.0)
        x = [3.0, 5.0, 6.0]
        full = LineLoad([0, 10.0], [0.0, 2.0]).eval_approx(beam, x)
        self.assertTrue(np.allclose(full[:, 1], 2.0))
        partial = LineLoad([2.0, 7.0], [0.0, 1.0]).eval_approx(beam, x)
        self.assertTrue(np.allclose(partial[:, 1], 1.0, atol=1e-2))
        outside = LineLoad([2.0, 7.0], [0.0, 1.0]).eval_approx(beam, [1.0, 8.5])
        self.assertTrue(np.allclose(outside[:, 1], 0.0, atol=1e-2))

    def test_plate_moment_load(self):
        Lx, Ly = 2.0, 3.0
        plate = NavierPlate((Lx, Ly), (100, 120), D=np.eye(3))
        points = np.array([[1.0, 1.5], [0.7, 0.4], [1.4, 2.5]])
        full = RectangleLoad([[0, 0], [Lx, Ly]], [0, 1.0, -2.0])
        values = full.eval_approx(plate, points)
        self.assertTrue(np.allclose(values[:, 1], 1.0, atol=2e-2))
        self.assertTrue(np.allclose(values[:, 2], -2.0, atol=4e-2))
        patch = RectangleLoad([[0.5, 0.7], [1.3, 1.9]], [0, 1.0, -2.0])
        values = patch.eval_approx(plate, [[0.9, 1.3], [1.7, 2.5]])
        self.assertTrue(np.allclose(values[:, 1], [1.0, 0.0], atol=2e-2))
        self.assertTrue(np.allclose(values[:, 2], [-2.0, 0.0], atol=4e-2))


class TestResultAssignment(SigmaEpsilonTestCase):
    def test_non_load_items_are_skipped(self):
        L = 10.0
        x = np.linspace(0, L, 5)
        loads = LoadGroup(
            meta="note",
            A=PointLoad(L / 3, [1.0, 0.0]),
            G=LoadGroup(info=1, B=PointLoad(L / 2, [1.0, 0.0])),
            C=PointLoad(L / 4, [0.0, 1.0]),
        )
        beam = NavierBeam(L, 20, EI=1.0)
        res = beam.linear_static_analysis(x, loads)
        self.assertEqual(sorted(res.keys()), ["A", "C", "G"])
        self.assertEqual(list(res["G"].keys()), ["B"])
        for address in (["A"], ["G", "B"], ["C"]):
            case = (
                loads[address[0]]
                if len(address) == 1
                else loads[address[0]][address[1]]
            )
            reference = beam.linear_static_analysis(x, LoadGroup(X=case))["X"]
            self.assertTrue(np.allclose(res[tuple(address)].values, reference.values))

    def test_cases_with_addresses(self):
        A, B = PointLoad(1.0, [1.0, 0.0]), PointLoad(2.0, [1.0, 0.0])
        loads = LoadGroup(meta="note", A=A, G=LoadGroup(B=B))
        items = list(loads.cases(return_address=True))
        self.assertEqual([address for address, _ in items], [["A"], ["G", "B"]])
        self.assertTrue(items[0][1] is A and items[1][1] is B)
        self.assertEqual(list(loads.cases()), [A, B])

    def test_no_load_cases(self):
        beam = NavierBeam(10.0, 20, EI=1.0)
        with self.assertRaises(ValueError):
            beam.linear_static_analysis([1.0], LoadGroup(meta="note"))
        plate = NavierPlate((2.0, 3.0), (10, 10), D=np.eye(3))
        with self.assertRaises(ValueError):
            plate.linear_static_analysis([[1.0, 1.0]], LoadGroup(meta="note"))


if __name__ == "__main__":
    unittest.main()
