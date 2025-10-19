from stat_test.init import *
from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd


class KCI(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.kci_ui = KCI_UInd()
        self.kci_ci = KCI_CInd()
        self.method = 'kci'

    def p_cal_func(self, Xs, Ys, condition_set):
        p = self.kci_ui.compute_pvalue(self.data[:, Xs], self.data[:, Ys])[0] if len(condition_set) == 0 else \
            self.kci_ci.compute_pvalue(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set])[0]
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
