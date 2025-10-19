import time
import logging
import numpy as np
import networkx as nx
from utils.tools import cumulative_calculate_mu, cumulative_calculate_var, random_rank
from collections import Counter
from copy import deepcopy
from algo.pc import pc
from scipy.stats import norm, wishart
import pandas as pd


class GibbsSampler():
    def __init__(self, data, burn_in=20, post_burn_in=100, **kwargs):
        """
        To see details: https://lib.stat.cmu.edu/aoas/107/
        :param data:
        :param kwargs:
        """
        self.S = None  # 初始协方差阵
        self.df0 = None  # 初始Wishart分布的自由度
        self.scale = None  # 初始Wishart分布的尺度矩阵
        try:
            self.data = pd.DataFrame(data)
        except Exception:
            raise "data must be a pandas DataFrame or numpy array!"
        self.n, self.p = data.shape
        self.burn_in = burn_in  # Gibbs预热数目
        self.post_burn_in = post_burn_in  # Gibbs预热后，采样数目
        self.NSCAN = burn_in + post_burn_in
        self._initialization()

    def _initialization(self):
        rank_matrix = self.data.apply(random_rank, axis=0)
        Z = norm.ppf(rank_matrix / (self.n + 1))  # rank and normalize
        Zfill = np.random.normal(size=(self.n, self.p))
        Z[np.isnan(self.data)] = Zfill[np.isnan(self.data)]  # 将空值为处替换为独立随机正态样本
        self.Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)  # 标准化
        self.S = np.cov(self.Z, rowvar=False)
        self.R = self.data.rank(method='dense')
        self.df0 = self.p + 1
        epsilon = 1e-6
        self.scale = np.eye(self.p, self.p) * epsilon

    def burn_forward(self):
        """
        单次更新correlation matrix, 一般在循环中被调用
        :return:
        """
        for j in np.random.permutation(self.p):
            sj_rvec_delj_element = np.delete(self.S[j, :], j)  # S的第j列删除第j个元素
            sj_cvec_delj_element = sj_rvec_delj_element.T  # S的第j行删除第j个元素
            s_mat_delj_rowcol = np.delete(np.delete(self.S, j, axis=0), j, axis=1)  # S删除第j行和第j列
            inv_s_mat_delj_rowcol = np.linalg.inv(s_mat_delj_rowcol)
            Sjc = sj_rvec_delj_element @ inv_s_mat_delj_rowcol
            sdj = np.sqrt(self.S[j, j] - sj_rvec_delj_element @ inv_s_mat_delj_rowcol @ sj_cvec_delj_element)
            muj = np.delete(self.Z, j, axis=1) @ Sjc.T

            # 1.在数据不为空的时候，更新Z的元素
            for r in np.sort(np.unique(self.R.dropna().values[:, j])):
                series = self.R.iloc[:, j]
                ir = series[pd.notna(series) & np.isclose(series, r)].index
                lb = np.max(self.Z[series < r, j], initial=-np.inf)
                ub = np.min(self.Z[series > r, j], initial=np.inf)
                u = np.random.uniform(norm.cdf(lb, loc=muj[ir], scale=sdj), norm.cdf(ub, loc=muj[ir], scale=sdj))
                self.Z[ir, j] = muj[ir] + sdj * norm.ppf(u)

            # 2.在数据为空的时候，更新Z的元素
            ir_nan = np.where(np.isnan(self.R.iloc[:, j]))
            self.Z[ir_nan, j] = np.random.normal(loc=muj[ir_nan], scale=sdj)

        # update S
        self.S = np.linalg.inv(
            wishart.rvs(scale=np.linalg.inv(self.scale * self.df0 + self.Z.T @ self.Z), df=self.df0 + self.n))
        # end of Gibbs sampling scheme
        # 计算协方差矩阵的对角线元素的标准差
        std_devs = np.sqrt(np.diag(self.S))
        # 计算相关矩阵
        correlation_matrix = self.S / np.outer(std_devs, std_devs)
        return correlation_matrix

    def sampling_cor_matrix_list(self):
        """
        用于采样整个correlation matrix list, 在correlation matrix的dimension较大时，容易造成内存爆炸
        :return:
        """
        cor_matrix_list = []
        for nscan in range(self.NSCAN):
            correlation_matrix = self.burn_forward()
            if nscan >= self.burn_in:
                cor_matrix_list.append(correlation_matrix)
        return cor_matrix_list


def gbs_sampling(data, config):
    algo_config = config.evaluation.algorithm
    test_config = config.test
    stable = config.evaluation.algorithm.copula_pc_kwargs.copula_pc_stable

    gs = GibbsSampler(data, **algo_config.copula_pc_kwargs.__dict__)
    p = data.shape[1]

    if stable:
        if hasattr(test_config, 'l'):
            assert test_config.l > 0, logging.warning('Sample number of correlation matrix must be positive')
        else:
            logging.warning('Sample number of correlation matrix is not specified, set to 20% of post_burn_in')
            test_config.l = int(.2 * gs.post_burn_in)
        step = gs.post_burn_in // test_config.l
        start = gs.NSCAN - 1 - (test_config.l - 1) * step

    # calculate n_hat
    expectation_Cij, var_Cij = np.zeros(int(p * (p - 1) / 2)), np.zeros(int(p * (p - 1) / 2))  # 计算期望和方差期望
    cor_hat = np.zeros((p, p))
    cor_matrix_list_equidistant = []
    # 采用增量式计算,节约内存
    for nscan in range(gs.NSCAN):
        cor_burn_forward = gs.burn_forward()
        if nscan >= gs.burn_in:
            # 增量式计算
            upper_tri = np.triu(cor_burn_forward, k=1)  # 取出上三角部分（不包括对角线）
            upper_tri_flat = upper_tri[upper_tri != 0].flatten()  # 将上三角部分拉成一维数组
            var_Cij = cumulative_calculate_var(var_Cij, expectation_Cij, upper_tri_flat,
                                               n=nscan)  # var_n -> var_{n+1}
            expectation_Cij = cumulative_calculate_mu(expectation_Cij, upper_tri_flat,
                                                      step=nscan)  # mu_n -> mu_{n+1}
            if not stable:
                cor_hat = cumulative_calculate_mu(cor_hat, cor_burn_forward, step=nscan)
            if stable and nscan in np.arange(start, gs.NSCAN, step):
                cor_matrix_list_equidistant.append(cor_burn_forward)
    v = (1 - expectation_Cij ** 2) ** 2 / var_Cij
    n_hat = np.mean(v)

    return cor_matrix_list_equidistant if stable else cor_hat, n_hat


def copula_pc_stable(data, uit, cit, alpha, beta, **kwargs):
    assert 0 < beta < 1, logging.warning('beta and  must be in [0, 1]', 'beta must be in [0, 1]')
    return _copula_pc_statble(data, uit, cit, alpha, beta)


def _copula_pc_statble(data, uit, cit, alpha, beta):
    pc_fit_list = []
    edge_count = []
    assert len(uit) == len(cit), logging.warning('In copula_pc, uit and cit must have the same length !')
    l = len(uit)
    for ut, ct in zip(uit, cit):
        pc_fit = pc(data, ut, ct, alpha)
        pc_fit.to_nx_graph()
        pc_fit_list.append(pc_fit)
        edges = list(pc_fit.nx_graph.edges)  # CPDAG
        edge_count.extend(edges)
    # 统计边的个数频率
    edge_count_counter = Counter(edge_count)
    edges_remain = {key: count / l for key, count in edge_count_counter.items() if count / l >= beta}
    # print(edges_remain)
    pc_fit.nx_graph = nx.DiGraph(edges_remain.keys())
    pc_fit.nx_skel = pc_fit.nx_graph.to_undirected()
    pc_fit.PC_elapsed = time.time() - ut.test_start_time
    return pc_fit


def copula_pc(data, uit, cit, alpha, **kwargs):
    return _copula_pc(data, uit, cit, alpha)


def _copula_pc(data, uit, cit, alpha):
    """

    :param alpha:
    :param data:
    :param l: 等距采样的cor_matrix个数
    :param beta: 边采样下阈值
    :param copula_pc_stable: 是否采用Stable版本的Copula PC
    :param kwargs:
    :return:
    """
    pc_fit = pc(data, uit, cit, alpha)
    pc_fit.PC_elapsed = time.time() - uit.test_start_time
    return pc_fit
