import logging

from stat_test.init import *


class D_Separation(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, true_dag, save_cache_cycle_seconds,
                 **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        '''
        Use d-separation as CI test, to ensure the correctness of constraint-based methods. (only used for tests)
        Parameters
        ----------
        data:   numpy.ndarray, just a placeholder, not used in D_Separation
        true_dag:   nx.DiGraph object, the true DAG
        '''
        self.true_dag = true_dag
        import networkx as nx;
        global nx
        self.method = 'd_separation'

    def p_cal_func(self, Xs, Ys, condition_set):
        p = float(nx.is_d_separator(self.true_dag, {Xs[0]}, {Ys[0]}, set(condition_set)))
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
