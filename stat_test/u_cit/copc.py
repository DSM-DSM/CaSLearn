import logging

from stat_test.u_cit.pcor import Fisherz


class Copula_Fisherz(Fisherz):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds,
                 effective_number, cor_matrix, t_start,**kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.sample_size = effective_number
        self.correlation_matrix = cor_matrix
        self.test_start_time = t_start
        self.method = 'copula_fisherz'
