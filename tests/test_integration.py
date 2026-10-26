import unittest
from unittest import mock

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.fourier import (
    NavierBeam,
    NavierPlate,
    LineLoad,
    RectangleLoad,
    DiskLoad,
)
from sigmaepsilon.solid.fourier.config import Config
from sigmaepsilon.solid.fourier.enums import IntegrationMethod
from sigmaepsilon.solid.fourier.mc import monte_carlo_samples
from sigmaepsilon.solid.fourier.quadrature import gauss_legendre
from sigmaepsilon.solid.fourier.utils import generate_random_points_in_rectangle
import sigmaepsilon.solid.fourier.integration as integration


def relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    return np.max(np.abs(values - reference)) / np.max(np.abs(reference))


def harmonics(shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    M, N = shape
    return np.repeat(np.arange(1, M + 1), N), np.tile(np.arange(1, N + 1), M)


class TestQuadrature(SigmaEpsilonTestCase):
    """
    Compares the coefficients integrated with Gauss-Legendre quadrature with
    closed-form ones.
    """

    size = Lx, Ly = (2.0, 3.0)

    def test_gauss_legendre_polynomial(self):
        x, w = gauss_legendre(1.0, 4.0, num_panels=3, order=4)
        self.assertAlmostEqual(np.sum(w), 3.0, places=14)
        self.assertAlmostEqual(np.dot(w, x**7), (4.0**8 - 1.0) / 8, places=8)

    def test_constant_rectangle_load(self):
        domain = [[0.5, 0.7], [1.3, 1.9]]
        for shape in [(10, 14), (60, 40)]:
            plate = NavierPlate(self.size, shape, D=np.eye(3))
            numeric = RectangleLoad(domain, ["1", "-2", "0.5"])
            const = RectangleLoad(domain, [1.0, -2.0, 0.5])
            self.assertLess(relative_error(numeric.rhs(plate), const.rhs(plate)), 1e-12)
            self.assertLess(
                relative_error(numeric.rhs0(plate), const.rhs0(plate)), 1e-12
            )

    def test_linear_rectangle_load(self):
        # the integral of x * sin(a * x) is sin(a * x) / a**2 - x * cos(a * x) / a
        (x0, y0), (x1, y1) = domain = [[0.5, 0.7], [1.3, 1.9]]
        shape = (12, 9)
        plate = NavierPlate(self.size, shape, D=np.eye(3))
        m, n = harmonics(shape)
        a, b = np.pi * m / self.Lx, np.pi * n / self.Ly
        Ix = lambda x: np.sin(a * x) / a**2 - x * np.cos(a * x) / a
        Iy = (np.cos(b * y0) - np.cos(b * y1)) / b
        reference = 4 / (self.Lx * self.Ly) * (Ix(x1) - Ix(x0)) * Iy
        rhs = RectangleLoad(domain, ["x", 0, 0]).rhs(plate)
        self.assertLess(relative_error(rhs[:, 0], reference), 1e-12)

    def test_trigonometric_load_convergence(self):
        # the results must not change if the rule is refined
        plate = NavierPlate(self.size, (20, 20), D=np.eye(3))
        load = RectangleLoad([[0.2, 0.3], [1.7, 2.6]], ["sin(3*x)*cos(2*y)", 0, 0])
        coarse = load.rhs(plate)
        with mock.patch.object(Config, "QUADRATURE_PANELS_PER_HALF_WAVE", 4):
            fine = load.rhs(plate)
        self.assertLess(relative_error(coarse, fine), 1e-12)

    def test_constant_disk_load(self):
        # the integral of sin(a * x) * sin(b * y) over a disk is
        # sin(a * xc) * sin(b * yc) * 2 * pi * R * J1(k * R) / k, k = sqrt(a**2 + b**2)
        from scipy.special import j1

        (xc, yc), R = center, R = (1.0, 1.5), 0.4
        for shape in [(10, 14), (50, 50)]:
            plate = NavierPlate(self.size, shape, D=np.eye(3))
            m, n = harmonics(shape)
            a, b = np.pi * m / self.Lx, np.pi * n / self.Ly
            k = np.hypot(a, b)
            reference = (
                4 / (self.Lx * self.Ly)
                * np.sin(a * xc)
                * np.sin(b * yc)
                * 2 * np.pi * R * j1(k * R) / k
            )  # fmt: skip
            rhs = DiskLoad((center, R), [1.0, 0, 0]).rhs(plate)
            self.assertLess(relative_error(rhs[:, 0], reference), 1e-12)

    def test_line_load_on_plate_along_y(self):
        x0, y0, y1 = 0.8, 0.5, 2.0
        shape = (10, 14)
        plate = NavierPlate(self.size, shape, D=np.eye(3))
        m, n = harmonics(shape)
        a, b = np.pi * m / self.Lx, np.pi * n / self.Ly
        c = 4 / (self.Lx * self.Ly)
        reference_f = c * np.sin(a * x0) * (np.cos(b * y0) - np.cos(b * y1)) / b
        reference_mx = c * np.sin(a * x0) * (np.sin(b * y1) - np.sin(b * y0)) / b
        rhs = LineLoad([[x0, y0], [x0, y1]], [1.0, 1.0, 0]).rhs(plate)
        self.assertLess(relative_error(rhs[:, 0], reference_f), 1e-12)
        self.assertLess(relative_error(rhs[:, 1], reference_mx), 1e-12)

    def test_line_load_on_beam(self):
        for N in [10, 500]:
            beam = NavierBeam(10.0, N, EI=1.0)
            numeric = LineLoad([2.0, 7.0], ["1", "1.5"])
            const = LineLoad([2.0, 7.0], [1.0, 1.5])
            self.assertLess(relative_error(numeric.rhs(beam), const.rhs(beam)), 1e-11)
            self.assertLess(relative_error(numeric.rhs0(beam), const.rhs0(beam)), 1e-12)

    def test_quadrature_is_deterministic(self):
        plate = NavierPlate(self.size, (8, 8), D=np.eye(3))
        load = DiskLoad(((1.0, 1.5), 0.4), ["x*y", "x", "y"])
        self.assertTrue(np.array_equal(load.rhs(plate), load.rhs(plate)))


class TestMonteCarlo(SigmaEpsilonTestCase):
    size = (2.0, 3.0)
    domain = [[0.0, 0.0], [2.0, 3.0]]

    def plate(self) -> NavierPlate:
        return NavierPlate(self.size, (4, 4), D=np.eye(3))

    def test_samples_cover_remainder(self):
        # a last batch of a single sample must not be dropped
        sampler = lambda n, rng: generate_random_points_in_rectangle(
            (0, 0), (1, 1), n, rng=rng
        )
        batches = list(monte_carlo_samples(sampler, 5001, 5000, 6.0, rng=0))
        self.assertEqual([len(w) for _, w in batches], [5000, 1])
        self.assertAlmostEqual(sum(np.sum(w) for _, w in batches), 6.0, places=12)

    def test_accuracy_with_one_extra_sample(self):
        plate = self.plate()
        mc = RectangleLoad(self.domain, ["1", 0, 0], num_mc=5001, rng=0)
        exact = RectangleLoad(self.domain, [1.0, 0, 0])
        # the first coefficient is 16 / pi**2, with all samples it is close
        self.assertAlmostEqual(mc.rhs(plate)[0, 0], exact.rhs(plate)[0, 0], delta=0.05)

    def test_reproducible_with_seed(self):
        plate = self.plate()
        a = RectangleLoad(self.domain, ["x", 0, 0], num_mc=1000, rng=42)
        b = RectangleLoad(self.domain, ["x", 0, 0], num_mc=1000, rng=42)
        self.assertTrue(np.array_equal(a.rhs(plate), b.rhs(plate)))
        self.assertTrue(np.array_equal(a.rhs(plate), a.rhs(plate)))

    def test_generator_advances(self):
        plate = self.plate()
        rng = np.random.default_rng(1)
        load = RectangleLoad(self.domain, ["x", 0, 0], num_mc=1000, rng=rng)
        self.assertFalse(np.array_equal(load.rhs(plate), load.rhs(plate)))

    def test_global_random_state_is_untouched(self):
        plate = self.plate()
        np.random.seed(3)
        state = np.random.get_state()[1].copy()
        RectangleLoad(self.domain, ["x", 0, 0], num_mc=1000).rhs(plate)
        self.assertTrue(np.array_equal(state, np.random.get_state()[1]))

    def test_config_seed(self):
        plate = self.plate()
        load = RectangleLoad(self.domain, ["x", 0, 0], num_mc=1000)
        with mock.patch.object(Config, "MC_SEED", 5):
            a, b = load.rhs(plate), load.rhs(plate)
        self.assertTrue(np.array_equal(a, b))

    def test_monte_carlo_vs_quadrature(self):
        plate = self.plate()
        values = ["x*y", "x", "y"]
        mc = RectangleLoad(self.domain, values, num_mc=200_000, rng=0)
        quad = RectangleLoad(self.domain, values)
        self.assertLess(relative_error(mc.rhs(plate), quad.rhs(plate)), 2e-2)
        self.assertLess(relative_error(mc.rhs0(plate), quad.rhs0(plate)), 2e-2)

    def test_beam_line_load_uses_beam_settings(self):
        beam = NavierBeam(10.0, 4, EI=1.0)
        load = LineLoad([2.0, 7.0], ["x", 0], integration="mc", rng=0)
        with mock.patch.object(
            integration, "monte_carlo_samples", wraps=monte_carlo_samples
        ) as spy:
            load.rhs(beam)
        _, num_samples, batch_size, _, _ = spy.call_args.args
        self.assertEqual(num_samples, Config.NUM_MC_SAMPLES_BEAM)
        self.assertEqual(batch_size, Config.MC_BATCH_SIZE_BEAM)


class TestIntegrationMethod(SigmaEpsilonTestCase):
    domain = [[0.0, 0.0], [1.0, 1.0]]

    def test_default_is_quadrature(self):
        load = RectangleLoad(self.domain, ["x", 0, 0])
        self.assertIs(load.integration, IntegrationMethod.QUADRATURE)

    def test_num_mc_implies_monte_carlo(self):
        load = RectangleLoad(self.domain, ["x", 0, 0], num_mc=100)
        self.assertIs(load.integration, IntegrationMethod.MONTE_CARLO)

    def test_explicit_method_wins(self):
        load = RectangleLoad(
            self.domain, ["x", 0, 0], num_mc=100, integration="quadrature"
        )
        self.assertIs(load.integration, IntegrationMethod.QUADRATURE)
        load.integration = "monte_carlo"
        self.assertIs(load.integration, IntegrationMethod.MONTE_CARLO)
        load.integration = IntegrationMethod.QUADRATURE
        self.assertIs(load.integration, IntegrationMethod.QUADRATURE)
        load.integration = None
        self.assertIs(load.integration, IntegrationMethod.MONTE_CARLO)

    def test_config_method(self):
        load = RectangleLoad(self.domain, ["x", 0, 0])
        with mock.patch.object(Config, "INTEGRATION_METHOD", "mc"):
            self.assertIs(load.integration, IntegrationMethod.MONTE_CARLO)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            RectangleLoad(self.domain, ["x", 0, 0], integration="simpson")
        with self.assertRaises(ValueError):
            IntegrationMethod.from_value(1)

    def test_invalid_num_mc(self):
        with self.assertRaises(ValueError):
            RectangleLoad(self.domain, ["x", 0, 0], num_mc=0)
        with self.assertRaises(TypeError):
            RectangleLoad(self.domain, ["x", 0, 0], num_mc=10.5)

    def test_invalid_load_value(self):
        plate = NavierPlate((1.0, 1.0), (2, 2), D=np.eye(3))
        with self.assertRaises(ValueError):
            RectangleLoad(self.domain, ["x", None, 0]).rhs(plate)


if __name__ == "__main__":
    unittest.main()
