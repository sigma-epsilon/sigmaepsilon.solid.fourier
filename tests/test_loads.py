import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from sigmaepsilon.solid.fourier.loads import NavierLoadError
from sigmaepsilon.solid.fourier import (
    NavierBeam,
    LoadGroup,
    PointLoad,
    LineLoad,
    RectangleLoad,
)


class TestBeamLoads(SigmaEpsilonTestCase):
    def test_load_error_smoke(self):
        NavierLoadError()

    def test_beam_loads_smoke(self):
        L = 1000.0  # geometry

        loads = LoadGroup(
            concentrated=LoadGroup(
                LC1=PointLoad(x=L / 2, v=[1.0, 0.0]),
                LC5=PointLoad(x=L / 2, v=[0.0, 1.0]),
            ),
            distributed=LoadGroup(
                LC2=LineLoad(x=[0, L], v=[1.0, 0.0]),
                LC6=LineLoad(x=[L / 2, L], v=[0.0, 1.0]),
                LC3=LineLoad(x=[L / 2, L], v=["x", 0]),
                LC4=LineLoad(x=[L / 2, L], v=[0, "x"]),
                LC7=LineLoad(x=[L / 2, L], v=["x", "x"]),
            ),
        )
        loads.lock()

        loads.cooperative = loads.cooperative

        loads.problem
        loads["concentrated"]["LC1"].problem

    def test_line_Load(self):
        L = 1000.0  # geometry
        w, h = 20.0, 80.0  # rectangular cross-section
        E = 210000.0  # material

        I = w * h**3 / 12
        EI = E * I
        
        loads = LoadGroup(
            LC1=LineLoad(x=[0, L], v=[1.0, 0.0]),
            LC2=LineLoad(x=[L / 2, L], v=[0.0, 1.0]),
            LC3=LineLoad(x=[L / 2, L], v=["x", 0]),
            LC4=LineLoad(x=[L / 2, L], v=[0, "x"]),
            LC5=LineLoad(x=[L / 2, L], v=["x", "x"]),
        )
        loads.lock()

        x = np.linspace(0, L, 2)
        beam = NavierBeam(L, 2, EI=EI)
        beam.solve(loads, x)


class TestPlateLoads(SigmaEpsilonTestCase):
    def test_plate_loads_smoke(self):
        Lx, Ly = (600.0, 800.0)

        loads = LoadGroup(
            LG1=LoadGroup(
                LC1=RectangleLoad(x=[[0, 0], [Lx, Ly]], v=[-0.1, 0, 0]),
                LC2=RectangleLoad(
                    x=[[Lx / 3, Ly / 2], [Lx / 2, 2 * Ly / 3]], v=[-1, 0, 0]
                ),
            ),
            LG2=LoadGroup(
                LC3=PointLoad(x=[Lx / 3, Ly / 2], v=[-100.0, 0, 0]),
                LC4=PointLoad(x=[2 * Lx / 3, Ly / 2], v=[100.0, 0, 0]),
            ),
        )
        loads.lock()

        loads.cooperative = loads.cooperative
        loads["LG1"]["LC1"].region


if __name__ == "__main__":
    unittest.main()
