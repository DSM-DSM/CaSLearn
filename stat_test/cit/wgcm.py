from stat_test.init import *


class WeightedGCM(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, regr_meth='gam',
                 weight_meth='sign', weight_num=7, beta=0.3, wgcm_nsim=499, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.est_kwarg, self.fix_kwarg = {}, {}
        self.fix_kwarg = {
            'regr.meth': regr_meth,
            'weight.meth': weight_meth,
            'weight.num': weight_num,
            'nsim': wgcm_nsim,
        }
        self.est_kwarg = {
            'regr.meth': regr_meth,
            'beta': beta,
            'nsim': wgcm_nsim
        }
        self.method = 'wgcm'

    def wgcm_test_r(self, X, Y, Z):
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr
        from statsmodels.stats.multitest import multipletests

        numpy2ri.activate()
        pandas2ri.activate()
        wGCM_r = importr('weightedGCM')

        p1 = wGCM_r.wgcm_est(X, Y, Z, **self.est_kwarg)[0]
        p2 = wGCM_r.wgcm_fix(X, Y, Z, **self.fix_kwarg)[0]
        # Bonferroni correction
        bonferroni_corrected = multipletests(pvals=[p1, p2], alpha=self.alpha, method='bonferroni')
        p = bonferroni_corrected[1][0]
        return p

    def p_cal_func(self, Xs, Ys, condition_set):
        p = self.wgcm_test_r(self.data[:, Xs], self.data[:, Ys]) if len(condition_set) == 0 \
            else self.wgcm_test_r(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set])
        return p

    def __call__(self, X, Y, condition_set):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
