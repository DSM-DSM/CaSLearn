from stat_test.init import *
from stat_test.cit.dgan.dgancit import dgancit


class DoubleGAN(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, batch_size=64,
                 dgan_kold=2, dgan_n_iter=1000, M=500, b=30, j=1000, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.dgan_kwargs = {
            'batch_size': batch_size,
            'k': dgan_kold,
            'n_iter': dgan_n_iter,
            'M': M,
            'b': b,
            'j': j,
            'device': device,
        }
        self.method = 'dgan'

    def p_cal_func(self, Xs, Ys, condition_set):
        p = dgancit(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set], **self.dgan_kwargs)
        logging.info(f'p:{p}')
        return p

    def __call__(self, X, Y, condition_set=None, **kwargs):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
