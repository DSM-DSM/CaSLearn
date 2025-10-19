from stat_test.init import *


class Lp(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds,rank=1000, rank_GP=200, J=5, p_norm=2,
                 mu=1e-10, optimizer=True, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.lp_kwargs = {
            'rank': rank,
            'rank_GP': rank_GP,
            'J': J,
            'p_norm': p_norm,
            'mu': mu,
            'alpha': alpha,
            'optimizer': optimizer
        }
        self.method = 'lp'

    def p_val_func(self, Xs, Ys, condition_set):
        from stat_test.cit.lp.lp_ci_test import test_asymptotic_ci

        x, y, z = self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set]
        res = test_asymptotic_ci(x, y, z, **self.lp_kwargs)
        p = res['pvalue']
        return p

    def __call__(self, X, Y, condition_set=None, **kwargs):
        return self.cache_management(X, Y, condition_set, self.p_val_func)
