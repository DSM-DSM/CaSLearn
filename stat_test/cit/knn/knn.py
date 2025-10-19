from stat_test.init import *


class KNN(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, classifier='xgb', knn_normalize=False,
                 shuffle_neighbors=5, sig_samples=200, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.method = 'knn'
        self.knn_kwargs = {
            'classifier': classifier,
            'normalize': knn_normalize,
            'shuffle_neighbors': shuffle_neighbors,
            'sig_samples': sig_samples,
            'device': self.device
        }

    def data_scaling(self, x, y, z):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scale_x = scaler.fit_transform(x)
        scale_y = scaler.fit_transform(y)
        scale_z = scaler.fit_transform(z)
        return scale_x, scale_y, scale_z

    def p_val_func(self, Xs, Ys, condition_set):
        from stat_test.cit.knn.nnlscit import lpcmicit

        x, y, z = self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set]
        x, y, z = self.data_scaling(x, y, z)
        p = lpcmicit(x, y, z, **self.knn_kwargs)
        return p

    def __call__(self, X: int, Y: int, condition_set=None, **kwargs):
        return self.cache_management(X, Y, condition_set, self.p_val_func)
