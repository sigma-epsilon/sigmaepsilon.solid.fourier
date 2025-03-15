from typing import Protocol, runtime_checkable, Iterable, Hashable, ClassVar
from numpy import ndarray

from sigmaepsilon.deepdict import DeepDict

from .result import LoadCaseResultLinStat
from .enums import MechanicalModelType


@runtime_checkable
class NavierProblemProtocol(Protocol):

    result_class: ClassVar[LoadCaseResultLinStat]

    @property
    def size(self) -> Iterable[float | int] | float | int: ...

    @property
    def shape(self, value) -> Iterable[int] | int: ...

    @property
    def model_type(self) -> MechanicalModelType: ...

    def linear_static_analysis(
        self, points, loads
    ) -> DeepDict[Hashable, DeepDict | LoadCaseResultLinStat]: ...


@runtime_checkable
class LoadCaseProtocol(Protocol):
    
    @property
    def domain(self): ...

    @domain.setter
    def domain(self, value): ...

    @property
    def value(self): ...

    @value.setter
    def value(self, value): ...

    def eval_approx(self, problem: NavierProblemProtocol, points: Iterable) -> ndarray:
        """Evaluates the Fourier series approximation of the load at the given points."""
        ...

    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        """Calculates the Fourier coefficients."""
        ...


@runtime_checkable
class LoadGroupProtocol(Protocol):
    def groups(self, *args, **kwargs) -> Iterable["LoadGroupProtocol"]: ...
    def cases(self, *args, **kwargs) -> Iterable[LoadCaseProtocol]: ...
