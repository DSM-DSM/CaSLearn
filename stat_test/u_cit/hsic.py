from stat_test.init import *


class HSIC(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.method = 'hsic'

    def p_cal_func(self, Xs, Ys, condition_set):
        from conditional_independence import hsic_test
        # 会出现inf
        # https://conditional-independence.readthedocs.io/en/latest/ci_tests/generated/conditional_independence.hsic_test.html
        pval = hsic_test(self.data, Xs, Ys)['p_value']
        return pval

    def __call__(self, X, Y, condition_set=None, **kwargs):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
