from stat_test.init import *


class Diffusion(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, stat='cmi',
                 repeat=100, sampling_model='ddpm', centralize=False, num_steps=1000, diffusion_seed=8888, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.method = 'diffusion'
        self.diffusion_kwargs = {
            'stat': stat,
            'repeat': repeat,
            'device': self.device,
            'centralize': centralize,
            'sampling_model': sampling_model,
            'num_steps': num_steps,
            'seed': diffusion_seed,
        }

    def diffusion_cit(self, x_idx: list, y_idx: list, z_idx: list):
        from stat_test.cit.diffusion.diffusion_crt import perform_diffusion_crt

        total_x, total_y, total_z = self.data[:, x_idx], self.data[:, y_idx], self.data[:, z_idx]
        n = round(total_x.shape[0] / 2)
        xxx = total_x[:n, :]
        yyy = total_y[:n, :]
        zzz = total_z[:n, :]

        xxx_crt = total_x[n:, :]
        yyy_crt = total_y[n:, :]
        zzz_crt = total_z[n:, :]
        pvalue = perform_diffusion_crt(xxx, yyy, zzz, xxx_crt, yyy_crt, zzz_crt, **self.diffusion_kwargs)
        return pvalue

    def p_val_func(self, Xs, Ys, condition_set):
        p = self.diffusion_cit(Xs, Ys, condition_set)
        return p

    def __call__(self, X, Y, condition_set=None, **kwargs):
        return self.cache_management(X, Y, condition_set, self.p_val_func)
