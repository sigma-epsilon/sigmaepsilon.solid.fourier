from abc import abstractmethod

from .result import LoadCaseResultLinStat
from .protocols import LoadGroupProtocol

__all__ = ["NavierProblem"]


class NavierProblem:
    """
    Base class for Navier problems. The sole reason of this class is to
    avoid circular referencing.
    """

    result_class = LoadCaseResultLinStat
    
    def __init__(self, *, loads: LoadGroupProtocol | None = None):
        self._loads = loads

    @property
    def loads(self) -> LoadGroupProtocol | None:
        """
        Returns the loads.
        """
        return self._loads
    
    @loads.setter
    def loads(self, value: LoadGroupProtocol | None):
        """
        Sets the loads.
        """
        self._loads = value
    
    def _postproc_linstat_load_case_result(self, data) -> LoadCaseResultLinStat:
        res = self.result_class(data, name="values")
        return res

    @abstractmethod
    def linear_static_analysis(self, *args, **kwargs): ...
