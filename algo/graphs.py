from causallearn.graph.GraphClass import CausalGraph
from typing import List


class CausalGraphPlus(CausalGraph):
    def __init__(self, no_of_var: int, node_names: List[str] | None = None):
        super().__init__(no_of_var, node_names)
        self.uitest_instance = None
        self.citest_instance = None
        self.uit_copied = False

    def set_ind_test(self, uit, cit):
        """Set the conditional independence test that will be used"""
        self.uitest_instance = uit
        self.citest_instance = cit

    def idep_test(self, i: int, j: int, S) -> float:
        """Define the conditional independence test"""
        assert i != j and not i in S and not j in S
        if len(S) > 0:
            if not self.uit_copied:
                self.citest_instance.pvalue_cache = self.uitest_instance.pvalue_cache
                self.uit_copied = True
            return self.citest_instance(i, j, S)
        else:
            return self.uitest_instance(i, j)

    def vpca_idep_test(self, i: int, j: int, S) -> float:
        assert i != j and not i in S and not j in S
        if len(S) > 0:
            return self.citest_instance(i, j, S)
        else:
            return self.uitest_instance(i, j)

