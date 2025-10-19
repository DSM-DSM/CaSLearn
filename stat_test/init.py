from causallearn.utils.cit import CIT_Base
from robustbase.stats import Qn
from collections.abc import Iterable
from scipy.stats import norm
from scipy.stats import spearmanr, kendalltau
import numpy as np
import logging
import networkx as nx
from math import log, sqrt
from utils.tools import dict2namespace

d_separation = 'd_separation'
fisherz = 'fisherz'
chisq = 'chisq'
mv_fisherz = 'mv_fisherz'
kci = 'kci'
gsq = 'gsq'
rruit = 'rruit'
classifier = 'classifier'
gcm = 'gcm'
spearman = 'spearman'
kendall = 'kendall'
robustQn = 'robustQn'
conditional_distance = 'conditional_distance'
knn = 'knn'
gan = 'gan'
dgan = 'dgan'
lp = 'lp'
diffusion = 'diffusion'
hsic = 'hsic'
wgcm = 'wgcm'
copc = 'copc'
pcor = 'pcor'
vt = 'vt'

UIT_METHODS = [kci, gcm, spearman, fisherz, kendall, robustQn, classifier, conditional_distance, rruit]
CIT_METHODS = [kci, gcm, spearman, fisherz, kendall, robustQn, classifier, conditional_distance, lp, gan, diffusion,
               knn, dgan, wgcm]

VE_UIT_METHODS = [d_separation]
VE_CIT_METHODS = [kci, gcm, spearman, fisherz, kendall, robustQn, classifier, conditional_distance, lp, gan, diffusion,
                  knn, dgan, wgcm]

# logging.info(f'Voting UIT:{UIT_METHODS}')
# logging.info(f'Voting CIT:{CIT_METHODS}')
# logging.info(f'Voting Edge UIT:{VE_UIT_METHODS}')
# logging.info(f'Voting Edge CIT:{VE_CIT_METHODS}')


class Test_Base(CIT_Base):
    def __init__(self, data, cache_path, device, use_cache=True, save_cache=True, alpha=0.05,
                 save_cache_cycle_seconds=30, **kwargs):
        super().__init__(data, cache_path)
        self.assert_input_data_is_valid()
        self.method = None
        self.SAVE_CACHE_CYCLE_SECONDS = save_cache_cycle_seconds  # "-1" means that saving all test results
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.device = device
        self.alpha = alpha

    def save_cache_key(self, cache_key, p):
        if cache_key not in self.pvalue_cache.keys():
            self.pvalue_cache[cache_key] = {}
        self.pvalue_cache[cache_key].update({self.method: p})

    def p_exist(self, cache_key):
        if cache_key in self.pvalue_cache.keys():
            return self.method in self.pvalue_cache[cache_key].keys()
        else:
            return False

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        def _stringize(ulist1, ulist2, clist):
            # ulist1, ulist2, clist: list of ints, sorted.
            _strlst = lambda lst: '.'.join(map(str, lst))
            return f'{_strlst(ulist1)};{_strlst(ulist2)}|{_strlst(clist)}' if len(clist) > 0 else \
                f'{_strlst(ulist1)};{_strlst(ulist2)}'

        METHODS_SUPPORTING_MULTIDIM_DATA = ["kci"]
        if condition_set is None: condition_set = []
        # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        condition_set = sorted(set(map(int, condition_set)))

        # usually, X and Y are 1-dimensional index (in constraint-based methods)
        if self.method not in METHODS_SUPPORTING_MULTIDIM_DATA:
            X, Y = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
            assert X not in condition_set and Y not in condition_set, "X, Y cannot be in condition_set."
            return [X], [Y], condition_set, _stringize([X], [Y], condition_set)

        # also to support multi-dimensional unconditional X, Y (usually in kernel-based tests)
        Xs = sorted(set(map(int, X))) if isinstance(X, Iterable) else [int(X)]  # sorted for comparison
        Ys = sorted(set(map(int, Y))) if isinstance(Y, Iterable) else [int(Y)]
        Xs, Ys = (Xs, Ys) if (Xs < Ys) else (Ys, Xs)
        assert len(set(Xs).intersection(condition_set)) == 0 and \
               len(set(Ys).intersection(condition_set)) == 0, "X, Y cannot be in condition_set."
        return Xs, Ys, condition_set, _stringize(Xs, Ys, condition_set)

    def cache_management(self, X, Y, condition_set, p_cal_func):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if self.p_exist(cache_key) and self.use_cache: return self.pvalue_cache[cache_key][self.method]
        p = p_cal_func(Xs, Ys, condition_set)
        self.save_cache_key(cache_key, p)
        if self.save_cache: self.save_to_local_cache()
        return p
