from enum import Enum, auto, unique


@unique
class MechanicalModelType(Enum):
    BERNOULLI_EULER_BEAM = auto()
    TIMOSHENKO_BEAM = auto()
    UFLYAND_MINDLIN_PLATE = auto()
    KIRCHHOFF_LOVE_PLATE = auto()
