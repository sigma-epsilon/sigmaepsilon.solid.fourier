from abc import abstractmethod

from .result import LoadCaseResultLinStat

__all__ = ["NavierProblem"]


class NavierProblem:
    """
    Base class for Navier problems. The sole reason of this class is to
    avoid circular referencing.
    """

    result_class = LoadCaseResultLinStat

    def _postproc_linstat_load_case_result(self, data) -> LoadCaseResultLinStat:
        components = self.__class__.result_class.postproc_components
        res = LoadCaseResultLinStat(data, components=components, name="values")
        return res

    @abstractmethod
    def solve(self, *args, **kwargs): ...
