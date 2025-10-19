from stat_test.init import *


class ConditionalDistance(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds,
                 n_replicates=100, num_bootstrap=199, num_thread=8, cdcov_seed=888, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.kwargs = kwargs
        self.conditional_distance_uit_kwargs, self.conditional_distance_cit_kwargs = {}, {}
        self.conditional_distance_uit_kwargs['R'] = n_replicates
        self.conditional_distance_cit_kwargs['num.bootstrap'] = num_bootstrap
        self.conditional_distance_cit_kwargs['num.thread'] = num_thread
        self.seed = cdcov_seed
        self.method = 'conditional_distance'

    def cdcit_r(self, x, y, z):
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        cdcsis_r = importr('cdcsis')
        res = cdcsis_r.cdcov_test(x, y, z, self.seed, **self.conditional_distance_cit_kwargs)
        return res.rx2('p.value')[0]

    def cduit_r(self, x, y):
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import r

        numpy2ri.activate()
        energy_r = importr('energy')
        r(f'set.seed({self.seed})')
        res = energy_r.dcov_test(x, y, **self.conditional_distance_uit_kwargs)
        return res.rx2('p.value')[0]

    def cduit_python(self, Xs, Ys, condition_set):
        from dcor.independence import distance_covariance_test

        res = distance_covariance_test(
            Xs,
            Ys,
            exponent=0.5,
            num_resamples=100
        )
        p = res.pvalue
        return p

    def p_cal_func(self, Xs, Ys, condition_set):
        p = self.cdcit_r(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set]) if len(condition_set) > 0 \
            else self.cduit_r(self.data[:, Xs], self.data[:, Ys])
        # logging.info(f'p-value: {p}')
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
