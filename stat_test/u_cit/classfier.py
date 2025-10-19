from stat_test.init import *


class Classifier(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds,
                 max_depths=[6, 10, 13], n_estimators=[100, 200, 300], colsample_bytrees=[0.8], nfold=5,
                 feature_selection=0, train_samp=-1, classifier_k=1, threshold=0.03, num_iter=20, nthread=8,
                 bootstrap=False, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.classifier_kwargs = {
            'max_depths': max_depths,
            'n_estimators': n_estimators,
            'colsample_bytrees': colsample_bytrees,
            'nfold': nfold,
            'feature_selection': feature_selection,
            'train_samp': train_samp,
            'k': classifier_k,
            'threshold': threshold,
            'num_iter': num_iter,
            'nthread': nthread,
            'bootstrap': bootstrap
        }
        self.method = 'classifier'

    def classifier_cit(self, X, Y, Z=None):
        from CCIT import CCIT

        pvalue = CCIT.CCIT(X, Y, Z, **self.classifier_kwargs)
        return pvalue

    def p_cal_func(self, Xs, Ys, condition_set):
        p = self.classifier_cit(self.data[:, Xs], self.data[:, Ys]) if len(condition_set) == 0 \
            else self.classifier_cit(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set])
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
