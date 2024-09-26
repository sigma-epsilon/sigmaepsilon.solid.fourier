import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.fourier import LoadGroup, PointLoad, LineLoad, NavierBeam


class TestBernoulliBeam(SigmaEpsilonTestCase):
    def test_bernoulli_beam_smoke(self):
        L = 1000.0  # geometry
        w, h = 20.0, 80.0  # rectangular cross-section
        E = 210000.0  # material

        I = w * h**3 / 12
        EI = E * I

        loads = LoadGroup(
            concentrated=LoadGroup(
                LC1=PointLoad(L / 2, [1.0, 0.0]),
                LC5=PointLoad(L / 2, [0.0, 1.0]),
            ),
            distributed=LoadGroup(
                LC2=LineLoad([0, L], [1.0, 0.0]),
                LC6=LineLoad([L / 2, L], [0.0, 1.0]),
                LC3=LineLoad([L / 2, L], ["x", 0]),
            ),
        )
        loads.lock()

        x = np.linspace(0, L, 2)

        beam = NavierBeam(L, 2, EI=EI)
        solution = beam.solve(loads, x)

        load_case_solution = solution["concentrated"]["LC1"]
        load_case_solution.data
        load_case_solution.values
        load_case_solution.name = load_case_solution.name
        load_case_solution.to_xarray()
        load_case_solution.to_pandas()

    def test_invalid_load(self):
        L, EI = 1000.0, 1.0
        beam = NavierBeam(L, 2, EI=EI)
        x = np.linspace(0, L, 2)
        with self.assertRaises(TypeError):
            beam.solve(None, x)


class TestTimoshenkoBeam(SigmaEpsilonTestCase):
    def test_timoshenko_beam_smoke(self):
        L = 1000.0  # geometry
        w, h = 20.0, 80.0  # rectangular cross-section
        E, nu = 210000.0, 0.25  # material

        I = w * h**3 / 12
        A = w * h
        EI = E * I
        G = E / (2 * (1 + nu))
        GA = G * A * 5 / 6

        loads = LoadGroup(
            concentrated=LoadGroup(
                LC1=PointLoad(L / 2, [1.0, 0.0]),
                LC5=PointLoad(L / 2, [0.0, 1.0]),
            ),
            distributed=LoadGroup(
                LC2=LineLoad([0, L], [1.0, 0.0]),
                LC6=LineLoad([L / 2, L], [0.0, 1.0]),
                LC3=LineLoad([L / 2, L], ["x", 0]),
            ),
        )
        loads.lock()

        x = np.linspace(0, L, 2)

        beam = NavierBeam(L, 2, EI=EI, GA=GA)
        beam.solve(loads, x)


if __name__ == "__main__":
    unittest.main()
