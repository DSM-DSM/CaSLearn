from stat_test.init import *


class GCM(Test_Base):
    def __init__(self, data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds,
                 regr_method='xgboost', gcm_nsim=499, **kwargs):
        super().__init__(data, cache_path, device, use_cache, save_cache, alpha, save_cache_cycle_seconds)
        self.gcm_kwargs = {}
        self.gcm_kwargs = {
            'alpha': alpha,
            'nsim': gcm_nsim,
            'regr.method': regr_method,
        }
        self.method = 'gcm'

    def gcm_test_r(self, X, Y, Z=None):
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        pandas2ri.activate()
        GCM_r = importr('GeneralisedCovarianceMeasure')

        if Z is None:
            res = GCM_r.gcm_test(X, Y, **self.gcm_kwargs)
        else:
            res = GCM_r.gcm_test(X, Y, Z, **self.gcm_kwargs)

        return res.rx2('p.value')[0]

    def gcm_test_python(self, X, Y, Z=None):
        gcm = GeneralizeCovarianceMeasure(X, Y, Z)
        p = gcm()
        return p

    def p_cal_func(self, Xs, Ys, condition_set):
        p = self.gcm_test_r(self.data[:, Xs], self.data[:, Ys]) if len(condition_set) == 0 \
            else self.gcm_test_r(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set])
        return p

    def __call__(self, X, Y, condition_set=None):
        return self.cache_management(X, Y, condition_set, self.p_cal_func)


class GeneralizeCovarianceMeasure:
    def __init__(self, X, Y, Z, **kwargs):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.alpha = kwargs.get('alpha', 0.05)
        self.regr_method = kwargs.get('regr_method', 'xgboost')
        self.regr_pars = kwargs.get('regr_pars', {})
        self.nsim = kwargs.get('nsim', 499)
        self.resid_XonZ = kwargs.get('resid_XonZ', None)
        self.resid_YonZ = kwargs.get('resid_YonZ', None)
        self.initialize_residuals()

    @staticmethod
    def train_xgboost(X, y, pars=None):
        from sklearn.model_selection import KFold
        from xgboost import XGBRegressor
        import numpy as np

        # 设置默认参数
        if pars is None:
            pars = {}
        pars.setdefault('nrounds', 50)
        pars.setdefault('max_depth', [1, 3, 4, 5, 6])
        pars.setdefault('CV_folds', 10)
        pars.setdefault('ncores', 1)
        pars.setdefault('early_stopping', 10)
        pars.setdefault('silent', True)

        n = len(y)

        # 如果 X 为空或没有特征列
        if X is None or X.shape[1] == 0:
            result = {
                'Yfit': np.full(n, np.mean(y)),
                'residuals': y - np.mean(y),
                'model': None,
                'df': None,
                'edf': None,
                'edf1': None,
                'p.values': None
            }
            return result

        X = np.asarray(X)

        # 如果启用了交叉验证
        if 'CV_folds' in pars and pars['CV_folds'] is not None:
            num_folds = pars['CV_folds']
            rmse = np.zeros((pars['nrounds'], len(pars['max_depth'])))
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)

            for j, max_depth in enumerate(pars['max_depth']):
                for i, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = XGBRegressor(
                        objective='reg:squarederror',
                        max_depth=max_depth,
                        n_estimators=pars['nrounds'],
                        learning_rate=0.1,
                        n_jobs=pars['ncores'],
                        verbosity=0 if pars['silent'] else 1,
                        early_stopping_rounds=pars['early_stopping']
                    )

                    # 训练模型并启用早停
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        verbose=False
                    )

                    # 获取评估历史
                    evals_result = model.evals_result()
                    test_rmse = evals_result['validation_1']['rmse']  # validation_1：测试集（或验证集）的评估结果。

                    # 填充不足的轮次
                    if len(test_rmse) < pars['nrounds']:
                        test_rmse.extend([np.inf] * (pars['nrounds'] - len(test_rmse)))

                    # 累加 RMSE
                    rmse[:, j] += test_rmse

            # 找到最佳参数
            mins = np.unravel_index(np.argmin(rmse), rmse.shape)
            if not pars['silent']:
                print("RMSE Matrix:")
                print(rmse)
                print("Best Parameters Index:", mins)
                if mins[0] == 0 or mins[0] == pars['nrounds'] - 1 or mins[1] == 0 or mins[1] == len(
                        pars['max_depth']) - 1:
                    print("Warning: Selected parameters are at the extreme values of the CV grid.")

            final_nrounds = mins[0] + 1  # 转换为 1-based 索引
            final_max_depth = pars['max_depth'][mins[1]]
        else:
            # 如果未启用交叉验证但提供了多个 max_depth
            if len(pars['max_depth']) > 1:
                raise ValueError("Providing a vector of parameters must be used with CV.")
            final_max_depth = pars['max_depth'][0]
            final_nrounds = pars['nrounds']

        # 使用最佳参数训练最终模型
        model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=final_max_depth,
            n_estimators=final_nrounds,
            learning_rate=0.1,
            n_jobs=pars['ncores'],
            verbosity=0 if pars['silent'] else 1
        )
        model.fit(X, y)

        # 预测和残差计算
        Yfit = model.predict(X)
        residuals = y - Yfit

        return residuals

    @staticmethod
    def train_gam(X, y, pars):
        from pygam import LinearGAM
        gam = LinearGAM(**pars).fit(X, y)
        return gam

    # @staticmethod
    # def train_krr(X, y, pars):
    #     from sklearn.gaussian_process import KernelRidge
    #
    #     krr = KernelRidge(**pars).fit(X, y)
    #     return krr

    def comp_resids(self, response):
        response = np.array(response).astype(np.float64)  # 将V转换为浮点数数组
        if self.regr_method == 'gam':
            residual = self.train_gam(self.Z, response, self.regr_pars)
            return residual

        elif self.regr_method == 'xgboost':
            residual = self.train_xgboost(self.Z, response, self.regr_pars)
            return residual

        # elif self.regr_method == 'kernel.ridge':
        #     model = train_krr(self.Z, V, self.regr_pars)
        #     return residual
        else:
            raise ValueError("Unsupported regression method.")

    def initialize_residuals(self):
        if self.Z is None:
            self.resid_XonZ = self.X if self.resid_XonZ is None else self.resid_XonZ
            self.resid_YonZ = self.Y if self.resid_YonZ is None else self.resid_YonZ
        else:
            if self.resid_XonZ is None:
                if self.X is None:
                    raise ValueError("Either X or resid_XonZ must be provided.")
                self.resid_XonZ = np.apply_along_axis(self.comp_resids, 0, self.X)
            if self.resid_YonZ is None:
                if self.Y is None:
                    raise ValueError("Either Y or resid_YonZ must be provided.")
                self.resid_YonZ = np.apply_along_axis(self.comp_resids, 0, self.Y)

    def __call__(self, *args, **kwargs):
        if self.resid_XonZ.shape[1] > 1 or self.resid_YonZ.shape[1] > 1:
            d_X, d_Y = np.array(self.resid_XonZ).shape[1], np.array(self.resid_YonZ).shape[1]
            nn = self.resid_XonZ.shape[0]
            x_component = np.repeat(self.resid_XonZ, d_Y, axis=0).reshape(1, nn * d_Y * d_X)
            y_component = self.resid_YonZ[:, [i for i in range(d_Y) for _ in range(d_X)]].reshape(1, nn * d_Y * d_X)
            R_mat = (x_component * y_component).reshape(nn, d_X * d_Y).T
            R_mat = R_mat / np.sqrt((np.mean(R_mat ** 2, axis=1) - np.mean(R_mat, axis=1) ** 2)).reshape(-1, 1)
            test_statistic = np.max(np.abs(np.mean(R_mat, axis=1))) * np.sqrt(nn)
            test_statistic_sim = np.max(np.abs(R_mat @ np.random.randn(nn * self.nsim).reshape(nn, self.nsim)),
                                        axis=0) / np.sqrt(nn)
            p_value = (np.sum(test_statistic_sim >= test_statistic) + 1) / (self.nsim + 1)
        else:
            nn = len(self.resid_XonZ) if not hasattr(self.resid_XonZ, "shape") else self.resid_XonZ.shape[0]
            R = self.resid_XonZ * self.resid_YonZ
            mean_R = np.mean(R)
            test_statistic = np.sqrt(nn) * mean_R / np.sqrt(np.mean(R ** 2) - mean_R ** 2)
            p_value = 2 * norm.sf(abs(test_statistic))

        return p_value
