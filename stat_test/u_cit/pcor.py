from stat_test.init import *


class Fisherz(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.correlation_matrix = np.corrcoef(data.T)
        self.method = 'fisherz'

    def p_cal_func(self, Xs, Ys, condition_set):
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(
            r)  # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)


class Spearmanz(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.assert_input_data_is_valid()
        corr_spearman, _ = spearmanr(data)
        self.correlation_matrix = 2 * np.sin(np.pi / 6 * corr_spearman)
        self.method = 'spearman'

    def p_cal_func(self, Xs, Ys, condition_set):
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                'Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(
            r)  # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)


class Kendallz(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.assert_input_data_is_valid()
        self.calcualte_correlation_matrix()
        self.method = 'kendall'

    def calcualte_correlation_matrix(self):
        n = self.data.shape[1]
        correlation_matrix = np.zeros((n, n))
        for i in range(n):
            correlation_matrix[i, :] = [kendalltau(self.data[:, i], self.data[:, j])[0] if i != j else 1 for j in
                                        range(n)]
        self.correlation_matrix = 2 * np.sin(np.pi / 6 * correlation_matrix)

    def p_cal_func(self, Xs, Ys, condition_set):
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                'Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(
            r)  # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)


class RobustQnz(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha,save_cache_cycle_seconds)
        self.assert_input_data_is_valid()
        self.calculate_Qn_corr_mat()
        self.method = 'robustQn'

    def calculate_corr_ij(self, v1, v2):
        std1, std2 = np.std(v1), np.std(v2)
        q1 = Qn(v1 / std1 + v2 / std2)
        q2 = Qn(v1 / std1 - v2 / std2)
        return (q1 ** 2 - q2 ** 2) / (q1 ** 2 + q2 ** 2)

    def calculate_Qn_corr_mat(self):
        n = self.data.shape[1]
        self.correlation_matrix = np.zeros((n, n))

        for i in range(n):
            self.correlation_matrix[i, :] = [
                self.calculate_corr_ij(self.data[:, i], self.data[:, j]) if i != j else 1 for j in range(n)]

    def p_cal_func(self, Xs, Ys, condition_set):
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                'Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(
            r)  # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)
