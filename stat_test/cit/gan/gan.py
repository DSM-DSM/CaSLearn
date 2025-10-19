from stat_test.cit.gan.gancit import gancit
from stat_test.init import *


class GAN(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, statistic="rdc", lamda=10,
                 gan_normalize=True, gan_n_iter=1000, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.gan_kwargs = {
            'statistic': statistic,
            "lamda": lamda,
            "normalize": gan_normalize,
            "n_iter": gan_n_iter,
            'device': self.device,
        }
        self.method = 'gan'

    def p_val_func(self, Xs, Ys, condition_set):
        x, y, z = self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set]
        p = gancit(x, y, z, **self.gan_kwargs)
        return p

    def __call__(self, X, Y, condition_set=None, **kwargs):
        return self.cache_management(X, Y, condition_set, self.p_val_func)
