import unittest

from numpy import ascontiguousarray as ascont

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.mesh.grid import grid

from sigmaepsilon.solid.material import MindlinPlateSection as Section
from sigmaepsilon.solid.material import (
    ElasticityTensor,
    LinearElasticMaterial,
    HuberMisesHenckyFailureCriterion_SP,
)
from sigmaepsilon.solid.material.utils import elastic_stiffness_matrix
from sigmaepsilon.solid.fourier import (
    NavierPlate,
    LoadGroup,
    PointLoad,
    RectangleLoad,
)


class TestKirchhoffPlate(SigmaEpsilonTestCase):
    def test_kirchhoff_plate_smoke(self):
        size = Lx, Ly = (600.0, 800.0)
        E = 2890.0
        nu = 0.2
        t = 25.0
        yield_strength = 2.0

        hooke = elastic_stiffness_matrix(E=E, NU=nu)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(hooke, frame=frame, tensorial=False)

        failure_model = HuberMisesHenckyFailureCriterion_SP(
            yield_strength=yield_strength
        )

        material = LinearElasticMaterial(
            stiffness=stiffness, failure_model=failure_model
        )

        section = Section(
            layers=[
                Section.Layer(material=material, thickness=t),
            ]
        )
        ABDS = section.elastic_stiffness_matrix()
        D, S = ascont(ABDS[:3, :3]), ascont(ABDS[3:, 3:])

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

        grid_shape = (30, 40)
        gridparams = {"size": size, "shape": grid_shape, "eshape": "Q4"}
        coords, _ = grid(**gridparams)

        plate = NavierPlate(size, (20, 20), D=D)
        plate.loads = plate.loads
        plate.linear_static_analysis(coords, loads)

    def test_invalid_load(self):
        size = (600.0, 800.0)
        E = 2890.0
        nu = 0.2
        t = 25.0
        yield_strength = 2.0

        hooke = elastic_stiffness_matrix(E=E, NU=nu)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(hooke, frame=frame, tensorial=False)

        failure_model = HuberMisesHenckyFailureCriterion_SP(
            yield_strength=yield_strength
        )

        material = LinearElasticMaterial(
            stiffness=stiffness, failure_model=failure_model
        )

        section = Section(
            layers=[
                Section.Layer(material=material, thickness=t),
            ]
        )
        ABDS = section.elastic_stiffness_matrix()
        D = ascont(ABDS[:3, :3])

        plate = NavierPlate(size, (20, 20), D=D)

        with self.assertRaises(TypeError):
            plate.linear_static_analysis(None, None)


class TestMindlinPlate(SigmaEpsilonTestCase):
    def test_mindlin_plate_smoke(self):
        size = Lx, Ly = (600.0, 800.0)
        E = 2890.0
        nu = 0.2
        t = 25.0
        yield_strength = 2.0

        hooke = elastic_stiffness_matrix(E=E, NU=nu)
        frame = ReferenceFrame(dim=3)
        stiffness = ElasticityTensor(hooke, frame=frame, tensorial=False)

        failure_model = HuberMisesHenckyFailureCriterion_SP(
            yield_strength=yield_strength
        )

        material = LinearElasticMaterial(
            stiffness=stiffness, failure_model=failure_model
        )

        section = Section(
            layers=[
                Section.Layer(material=material, thickness=t),
            ]
        )
        ABDS = section.elastic_stiffness_matrix()
        D, S = ascont(ABDS[:3, :3]), ascont(ABDS[3:, 3:])

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

        grid_shape = (30, 40)
        gridparams = {"size": size, "shape": grid_shape, "eshape": "Q4"}
        coords, _ = grid(**gridparams)

        plate = NavierPlate(size, (20, 20), D=D, S=S)
        plate.linear_static_analysis(coords, loads)


if __name__ == "__main__":
    unittest.main()
