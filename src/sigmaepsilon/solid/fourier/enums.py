from enum import Enum, auto


class MechanicalModelType(Enum):
    BERNOULLI_EULER_BEAM = auto()
    TIMOSHENKO_BEAM = auto()
    UFLYAND_MINDLIN_PLATE = auto()
    KIRCHHOFF_LOVE_PLATE = auto()
