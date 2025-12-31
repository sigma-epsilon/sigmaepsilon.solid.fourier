from enum import Enum, auto, unique


@unique
class MechanicalModelType(Enum):
    """Enumeration of mechanical model types."""

    BERNOULLI_EULER_BEAM = auto()
    TIMOSHENKO_BEAM = auto()
    UFLYAND_MINDLIN_PLATE = auto()
    KIRCHHOFF_LOVE_PLATE = auto()

    @property
    def is_1d(self) -> bool:
        """Returns True if the model type is one-dimensional (beam)."""
        return self in {
            MechanicalModelType.BERNOULLI_EULER_BEAM,
            MechanicalModelType.TIMOSHENKO_BEAM,
        }

    @property
    def is_2d(self) -> bool:
        """Returns True if the model type is two-dimensional (plate)."""
        return self in {
            MechanicalModelType.UFLYAND_MINDLIN_PLATE,
            MechanicalModelType.KIRCHHOFF_LOVE_PLATE,
        }
