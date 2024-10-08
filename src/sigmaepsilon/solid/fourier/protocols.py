from typing import Protocol, runtime_checkable, Iterable, Hashable
from numpy import ndarray

from sigmaepsilon.deepdict import DeepDict

from .result import LoadCaseResultLinStat


@runtime_checkable
class NavierProblemProtocol(Protocol):
    def linear_static_analysis(
        self, points, loads
    ) -> DeepDict[Hashable, DeepDict | LoadCaseResultLinStat]: ...


@runtime_checkable
class LoadCaseProtocol(Protocol):
    def rhs(self, problem: NavierProblemProtocol) -> ndarray: ...


@runtime_checkable
class LoadGroupProtocol(Protocol):
    def groups(self, *args, **kwargs) -> Iterable["LoadGroupProtocol"]: ...
    def cases(self, *args, **kwargs) -> Iterable[LoadCaseProtocol]: ...
