import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from sigmaepsilon.solid.fourier import (
    NavierPlate,
    NavierBeam,
    LoadGroup,
    PointLoad,
    LineLoad,
    RectangleLoad,
    DiskLoad,
)


class TestBeamLoads(SigmaEpsilonTestCase):

    def test_beam_loads_smoke(self):
        L = 1000.0  # geometry

        loads = LoadGroup(
            concentrated=LoadGroup(
                LC1=PointLoad(L / 2, [1.0, 0.0]),
                LC5=PointLoad(L / 2, [0.0, 1.0]),
            ),
            distributed=LoadGroup(
                LC2=LineLoad([0, L], [1.0, 0.0]),
                LC6=LineLoad([L / 2, L], [0.0, 1.0]),
                LC3=LineLoad([L / 2, L], ["x", 0]),
                LC4=LineLoad([L / 2, L], [0, "x"]),
                LC7=LineLoad([L / 2, L], ["x", "x"]),
            ),
        )
        loads.lock()

        loads.cooperative = loads.cooperative
        loads.groups()

        load_case = loads["concentrated", "LC1"]
        load_case.domain = load_case.domain
        load_case.value = load_case.value

    def test_line_Load(self):
        L = 1000.0  # geometry
        w, h = 20.0, 80.0  # rectangular cross-section
        E = 210000.0  # material

        I = w * h**3 / 12
        EI = E * I

        loads = LoadGroup(
            LC1=LineLoad([0, L], [1.0, 0.0]),
            LC2=LineLoad([L / 2, L], [0.0, 1.0]),
            LC3=LineLoad([L / 2, L], ["x", 0], num_mc=100),
            LC4=LineLoad([L / 2, L], [0, "x"], num_mc=100),
            LC5=LineLoad([L / 2, L], ["x", "x"], num_mc=100),
        )
        loads.lock()

        x = np.linspace(0, L, 2)
        beam = NavierBeam(L, 2, EI=EI)
        beam.linear_static_analysis(x, loads)

    def test_eval_smoke(self):
        L = 1000.0  # geometry
        w, h = 20.0, 80.0  # rectangular cross-section
        E = 210000.0  # material

        I = w * h**3 / 12
        EI = E * I

        loads = LoadGroup(
            LC1=LineLoad([0, L], [1.0, 0.0]),
            LC2=PointLoad(L / 3, [1.0, 0.0]),
        )
        loads.lock()

        x = np.linspace(0, L, 2)
        beam = NavierBeam(L, 2, EI=EI)

        for case in loads.cases():
            case.eval_approx(beam, x)


class TestPlateLoads(SigmaEpsilonTestCase):
    def test_plate_loads_smoke(self):
        Lx, Ly = (600.0, 800.0)

        loads = LoadGroup(
            LG1=LoadGroup(
                LC1=RectangleLoad([[0, 0], [Lx, Ly]], [-0.1, 0, 0]),
                LC2=RectangleLoad([[Lx / 3, Ly / 2], [Lx / 2, 2 * Ly / 3]], [-1, 0, 0]),
            ),
            LG2=LoadGroup(
                LC3=PointLoad([Lx / 3, Ly / 2], [-100.0, 0, 0]),
                LC4=PointLoad([2 * Lx / 3, Ly / 2], [100.0, 0, 0]),
            ),
        )
        loads.lock()

        loads.cooperative = loads.cooperative
        loads["LG1"]["LC1"].region

    def test_rectangle_load_coeff_shape(self):
        length_X, length_Y = 10.0, 20.0
        number_of_modes_X, number_of_modes_Y = 10, 10
        bending_stiffness = np.eye(3)  # just to have some data
        kirchhoff_plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )
        load_case = RectangleLoad([[0, 0], [length_X, length_Y]], [-0.1, 0, 0])
        self.assertEqual(
            load_case.rhs(kirchhoff_plate).shape,
            (number_of_modes_X * number_of_modes_Y, 3),
        )

    def test_eval_smoke(self):
        length_X, length_Y = 10.0, 20.0
        number_of_modes_X, number_of_modes_Y = 10, 10
        bending_stiffness = np.eye(3)  # just to have some data
        kirchhoff_plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )
        load_cases = [
            RectangleLoad([[0, 0], [length_X, length_Y]], [-0.1, 0, 0]),
            PointLoad([length_X / 3, length_Y / 2], [-100.0, 0, 0]),
        ]
        points = np.array([[length_X / 3, length_Y / 2]])
        for case in load_cases:
            case.eval_approx(kirchhoff_plate, points)
            
    def test_disk_load(self):
        length_X, length_Y = 10.0, 20.0
        number_of_modes_X, number_of_modes_Y = 10, 10
        bending_stiffness = np.eye(3)  # just to have some data
        kirchhoff_plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )
        load_case = DiskLoad(((length_X / 2, length_Y / 2), 100), [10, 0, 0], num_mc=100)
        load_case.rhs(kirchhoff_plate)
        
    def test_line_load_2d(self):
        length_X, length_Y = (600.0, 800.0)

        # solution parameters
        number_of_modes_X = 10
        number_of_modes_Y = 10

        # material properties
        bending_stiffness = np.eye(3)

        # set up domains
        beam = NavierBeam(600, 2, EI=1000)
        plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )

        load_case = LineLoad(
            [[length_X / 3, length_Y / 3], [2 * length_X / 3, 2 * length_Y / 3]],
            [10, 0, 0],
            num_mc=100,
        )

        load_case.rhs(plate)
        
        with self.assertRaises(AssertionError):
            load_case.rhs(beam)
    
    def test_line_load_2d_invalid_values_shape_error(self):
        length_X, length_Y = (600.0, 800.0)

        # solution parameters
        number_of_modes_X = 10
        number_of_modes_Y = 10

        # material properties
        bending_stiffness = np.eye(3)

        # set up domains
        plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )
        
        load_case = LineLoad(
            [[length_X / 3, length_Y / 3], [2 * length_X / 3, 2 * length_Y / 3]],
            [10, 0],
            num_mc=100,
        )
        
        with self.assertRaises(AssertionError):
            load_case.rhs(plate)
        
    def test_line_load_2d_sym(self):
        length_X, length_Y = (600.0, 800.0)

        # solution parameters
        number_of_modes_X = 10
        number_of_modes_Y = 10

        # material properties
        bending_stiffness = np.eye(3)

        # set up plate
        plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )

        load_case = LineLoad(
            [[length_X / 3, length_Y / 3], [2 * length_X / 3, 2 * length_Y / 3]],
            ["(x+y)/100", 0, 0],
            num_mc=100,
        )

        load_case.rhs(plate)
        
    def test_rectangle_load_sym(self):
        length_X, length_Y = (600.0, 800.0)

        # solution parameters
        number_of_modes_X = 10
        number_of_modes_Y = 10

        # material properties
        bending_stiffness = np.eye(3)

        # set up plate
        plate = NavierPlate(
            (length_X, length_Y),
            (number_of_modes_X, number_of_modes_Y),
            D=bending_stiffness,
        )

        load_case = RectangleLoad(
            [[length_X / 4, length_Y / 4], [3*length_X / 4, 3*length_Y / 4]],
            ["(x+y)/100", 0, 0],
        )

        load_case.rhs(plate)


if __name__ == "__main__":
    unittest.main()
