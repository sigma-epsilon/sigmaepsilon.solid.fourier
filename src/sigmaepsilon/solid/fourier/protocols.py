from typing import Protocol, runtime_checkable, Iterable, Hashable, ClassVar
from numpy import ndarray

from sigmaepsilon.deepdict import DeepDict

from .result import LoadCaseResultLinStat
from .enums import MechanicalModelType


@runtime_checkable
class NavierProblemProtocol(Protocol):

    result_class: ClassVar[LoadCaseResultLinStat]

    @property
    def size(self) -> Iterable[float | int] | float | int:
        pass

    @property
    def shape(self, value) -> Iterable[int] | int:
        pass

    @property
    def model_type(self) -> MechanicalModelType:
        pass

    def linear_static_analysis(
        self, points, loads
    ) -> DeepDict[Hashable, DeepDict | LoadCaseResultLinStat]:
        pass


@runtime_checkable
class LoadCaseProtocol(Protocol):
    
    @property
    def domain(self):
        pass

    @domain.setter
    def domain(self, value):
        pass

    @property
    def value(self):
        pass

    @value.setter
    def value(self, value):
        pass

    def eval_approx(self, problem: NavierProblemProtocol, points: Iterable) -> ndarray:
        """Evaluates the Fourier series approximation of the load at the given points."""
        pass

    def rhs(self, problem: NavierProblemProtocol) -> ndarray:
        """Calculates the Fourier coefficients."""
        pass


@runtime_checkable
class LoadGroupProtocol(Protocol):
    def groups(self, *args, **kwargs) -> Iterable["LoadGroupProtocol"]:
        pass
    
    def cases(self, *args, **kwargs) -> Iterable[LoadCaseProtocol]:
        pass
