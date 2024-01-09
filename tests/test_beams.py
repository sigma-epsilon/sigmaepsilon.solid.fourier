import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.solid.fourier import LoadGroup, PointLoad, LineLoad, NavierBeam


class TestBeams(SigmaEpsilonTestCase):
    def test_bernoulli_beam(self):
        L = 1000.0  # geometry
        w, h = 20.0, 80.0  # rectangular cross-section
        E = 210000.0  # material

        I = w * h**3 / 12
        EI = E * I

        loads = LoadGroup(
            concentrated=LoadGroup(
                LC1=PointLoad(x=L / 2, v=[1.0, 0.0]),
                LC5=PointLoad(x=L / 2, v=[0.0, 1.0]),
            ),
            distributed=LoadGroup(
                LC2=LineLoad(x=[0, L], v=[1.0, 0.0]),
                LC6=LineLoad(x=[L / 2, L], v=[0.0, 1.0]),
                LC3=LineLoad(x=[L / 2, L], v=["x", 0]),
            ),
        )
        loads.lock()

        x = np.linspace(0, L, 2)

        beam = NavierBeam(L, 2, EI=EI)
        beam.solve(loads, x)

    def test_timoshenko_beam(self):
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
                LC1=PointLoad(x=L / 2, v=[1.0, 0.0]),
                LC5=PointLoad(x=L / 2, v=[0.0, 1.0]),
            ),
            distributed=LoadGroup(
                LC2=LineLoad(x=[0, L], v=[1.0, 0.0]),
                LC6=LineLoad(x=[L / 2, L], v=[0.0, 1.0]),
                LC3=LineLoad(x=[L / 2, L], v=["x", 0]),
            ),
        )
        loads.lock()

        x = np.linspace(0, L, 2)

        beam = NavierBeam(L, 2, EI=EI, GA=GA)
        beam.solve(loads, x)


if __name__ == "__main__":
    unittest.main()
