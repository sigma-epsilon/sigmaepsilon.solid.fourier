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


@unique
class IntegrationMethod(Enum):
    """Enumeration of the numerical integration methods of load coefficients."""

    QUADRATURE = "quadrature"
    MONTE_CARLO = "mc"

    @classmethod
    def from_value(cls, value: "str | IntegrationMethod") -> "IntegrationMethod":
        """
        Returns the method from an instance or a string. Besides the values of
        the members, "monte_carlo" is accepted for Monte Carlo integration.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.lower()
            if key == "monte_carlo":
                return cls.MONTE_CARLO
            for method in cls:
                if method.value == key:
                    return method
        raise ValueError(
            f"Invalid integration method {value!r}. "
            f"Valid options are {[m.value for m in cls]}."
        )
