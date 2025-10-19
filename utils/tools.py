import hashlib
import logging
import pandas as pd
from copy import deepcopy
import numpy as np
from cdt.metrics import SHD
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.DAG2CPDAG import dag2cpdag
import argparse
from argparse import Namespace
import networkx as nx
import yaml
import os
import json


def is_empty_json(file_path):
    """判断是否为内容为空的JSON文件"""
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return True
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果是空对象或空数组，也视为“空”
            if data == {} or data == []:
                return True
    except json.JSONDecodeError:
        # JSON 格式错误的文件不认为是“空”，但可以标记为损坏
        return False
    return False


def find_and_delete_invalid_json_cache(directory, file_name_list):
    """
    查找并删除指定目录下的空JSON文件。
    - dry_run=True 表示只显示要删除的文件，不实际删除
    """
    import codecs
    from json.decoder import JSONDecodeError
    logging.info("Empty file scanning done! We find empty file(s) and Delete them.")
    for file_name in file_name_list:
        full_file_path = os.path.join(directory, file_name)
        if full_file_path.endswith('.json'):
            if is_empty_json(full_file_path):
                os.remove(full_file_path)
                logging.info(f"[Deleted] {full_file_path}")
        if os.path.exists(full_file_path):
            try:
                with codecs.open(full_file_path, 'r') as fin:
                    pvalue_cache = json.load(fin)
            except JSONDecodeError:
                os.remove(full_file_path)
                os.makedirs(os.path.dirname(full_file_path), exist_ok=True)


def dump_yaml(path, config: Namespace):
    with open(path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def construct_namespace(loader, node):
    data = loader.construct_mapping(node)
    return Namespace(**data)


yaml.add_constructor('tag:yaml.org,2002:python/object:argparse.Namespace', construct_namespace)


def dict2namespace(config):
    """
    Convert dictionary to namespace recursively.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(namespace):
    """Recursively convert Namespace objects to dictionaries."""
    result = {}
    for key, val in namespace.__dict__.items():
        if isinstance(val, type(namespace)):  # 如果仍然是 Namespace
            result[key] = namespace2dict(val)
        else:
            result[key] = val
    return result


def nx_Digraph2causal_learn_CPDAG(graph: nx.DiGraph):
    """
    graph(nx.DiGraph) -> causal_learn_general_graph(causal_learn.Dag) -> gt_cpdag_adjm(np.ndarray) -> nx_cpdag(nx.DiGraph))
    :param graph:
    :return:
    """
    causal_learn_general_graph = adjacency_matrix_to_general_graph(nx.to_numpy_array(graph))
    gt_cpdag_adjm = dag2cpdag(causal_learn_general_graph).graph
    gt_cpdag_adjm[gt_cpdag_adjm == 1] = 0
    gt_cpdag_adjm[gt_cpdag_adjm == -1] = 1
    nx_cpdag = nx.from_numpy_array(gt_cpdag_adjm, create_using=nx.DiGraph)
    return nx_cpdag


def adjacency_matrix_to_general_graph(adj_matrix):
    n_nodes = adj_matrix.shape[0]
    nodes = [GraphNode(f"X{i}") for i in range(n_nodes)]
    graph = Dag(nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i][j] == 1:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                graph.add_edge(edge)
    return graph


def metric_calculation(pc_fit, graph, print_info=False):
    try:
        pc_fit.to_nx_graph()  # pc_fit.nx_graph(nx.DiGraph)
        pc_fit.to_nx_skeleton()
    except AttributeError:
        print(pc_fit)
    graph_cpdag = nx_Digraph2causal_learn_CPDAG(graph)
    shd = SHD(graph_cpdag, pc_fit.nx_graph, double_for_anticausal=False)
    shd_anti = SHD(graph_cpdag, pc_fit.nx_graph, double_for_anticausal=True)
    normalized_shd = shd / len(graph_cpdag.edges)
    normalized_shd_anti = shd_anti / len(graph_cpdag.edges)

    p = len(graph.nodes)
    e1 = set(graph.edges())
    e2 = set(pc_fit.nx_graph.edges())
    e1, e2 = e1 | set([(y, x) for x, y in e1]), e2 | set([(y, x) for x, y in e2])

    ### evaluate TPR and FPR w.r.t. skeleton
    # turn edges of Digraph and CPDAG to undirected
    common_edges = e1 & e2
    TP, FN, FP = len(common_edges) / 2, len(e1) / 2 - len(common_edges) / 2, len(e2) / 2 - len(common_edges) / 2
    TN = p * (p - 1) / 2 - (TP + FN + FP)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    metric_dict = {
        'SHD': shd,
        'SHD Anti': shd_anti,
        'Normalized SHD': normalized_shd,
        'Normalized SHD Anti': normalized_shd_anti,
        'time_spent': pc_fit.PC_elapsed,
        'TPR': TPR,
        'FPR': FPR,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'F1': F1,
        'Accuracy': Accuracy,
    }
    if print_info:
        print(f"gt DAG:{graph.edges()}")  # DAG
        print(f"gt CPDAG:{graph_cpdag.edges()}")  # (CP)DAG
        print(f"estimated CPDAG:{pc_fit.nx_graph.edges()}")  # (CP)DAG
        print(f"estimated skeleton:{pc_fit.nx_skel.edges()}")  # UnDAG
        print(metric_dict)
        print('\n')
    return metric_dict


def cumulative_calculate_mu(mu, x_new, step):
    """
    用于均值的增量式计算
    :param mu:
    :param x_new:
    :param step:
    :return:
    """
    if step == 0:
        return x_new
    else:
        return 1 / (step + 1) * x_new + mu * step / (step + 1)


def cumulative_calculate_var(var, mu, x_new, n):
    """
    用于方差的增量式计算
    :param var:
    :param mu:
    :param x_new:
    :param n:
    :return:
    """
    if n < 1:
        return np.zeros(x_new.shape)
    elif n == 1:
        variances = np.array([x ** 2 + y ** 2 - (x + y) ** 2 / 2 for x, y in zip(mu, x_new)])
        return variances
    else:
        delta = x_new - mu
        return (n - 1) / n * var + delta ** 2 / (n + 1)


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    seconds = seconds % 60

    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def res_print(v, name):
    mean = round(np.mean(v), 3)
    # 默认情况下，numpy使用的是总体标准差计算。ddof=0保持这一行为以匹配R中的sd函数。
    std = round(np.std(v, ddof=0), 3)
    mean_std = f"{mean}({std})"
    res = f"{name}:{mean_std}"
    print(res)
    globals()['mean_std'] = mean_std  # 在Python中，我们通过globals()来访问全局命名空间，模拟R中的assign行为。


def compute_hash(config_dict):
    config_json = json.dumps({k: v for k, v in config_dict.items()}, sort_keys=True, indent=4, ensure_ascii=False)
    return hashlib.sha256(config_json.encode('utf-8')).hexdigest()


def format_data(df, metric):
    # 初始化结果 DataFrame
    result = []

    from itertools import product
    for sample_size, points, degree in product(df['n_sample'].unique(), df['n_point'].unique(),
                                               df['exp_degree'].unique()):
        subset = df[(df['n_sample'] == sample_size) & (df['n_point'] == points) & (df['exp_degree'] == degree)]
        mechanism_l, noise_type_l, alpha_l = subset['data_gen_mechanism'].unique(), subset['noise_type'].unique(), \
            subset['alpha'].unique()
        new_row = {"SampleSize": sample_size, "NPoint": points, "Degree": degree}
        for mechanism, noise_type, alpha in product(mechanism_l, noise_type_l, alpha_l):
            row_key = f"{mechanism}_{noise_type}"
            # 构建新行
            v = \
                subset[(subset['data_gen_mechanism'] == mechanism) & (subset['noise_type'] == noise_type)][
                    metric].values[0]
            new_row.update({'alpha': alpha, row_key: v})
        result.append(new_row)
    # 转换为 DataFrame
    res_df = pd.DataFrame(result)
    return res_df


def random_rank(series):
    # 获取排序后的索引
    series_dc = deepcopy(series)
    series_dc[np.isnan(series_dc)] = np.max(series_dc) + 1
    ranks = series_dc.rank(method='first', ascending=True)
    return ranks


def comp_signal2noise_ratio(signal, noise, metric):
    sig_cov = np.cov(signal.T)
    noise_cov = np.cov(noise.T)
    if metric == 'tr':
        noise_tr = np.trace(noise_cov)
        sig_tr = np.trace(sig_cov)
        sig2noise = sig_tr / (noise_tr - sig_tr)
        return sig2noise, sig_tr, noise_tr
    elif metric == 'eigenvalue':
        sig_max_eigenvalues = np.max(np.linalg.eigvals(sig_cov))
        noise_max_eigenvalues = np.max(np.linalg.eigvals(noise_cov))
        sig2noise = sig_max_eigenvalues / (noise_max_eigenvalues - sig_max_eigenvalues)
        return sig2noise, sig_max_eigenvalues, noise_max_eigenvalues
    else:
        logging.error("When compute 'signal to mpise ratio', metric must be 'tr' or 'eigenvalue'")
        return None, None, None


def normalize_to_neg1_1(x):
    x_min = np.min(x)
    x_max = np.max(x)

    # 防止除以0（当所有元素相等时）
    if x_min == x_max:
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min) * 2 - 1


def extract_directed_and_undirected_edges(digraph):
    directed_edges = []
    undirected_edges = set()  # 使用集合避免重复

    # 遍历所有有向边
    for u, v in digraph.edges():
        # 跳过已处理的双向边
        if (u, v) in undirected_edges or (v, u) in undirected_edges:
            continue

        # 检查是否存在反向边
        if digraph.has_edge(v, u):
            # 双向边：存储为无序对（无向边）
            undirected_edges.add((min(u, v), max(u, v)))
        else:
            # 单向边
            directed_edges.append((u, v))

    return directed_edges, list(undirected_edges)


class TemporaryRandomSeed:
    def __init__(self, seed):
        self.seed = seed
        self.np_original_state = np.random.get_state()
        # self.random_original_state = random.getstate()

    def __enter__(self):
        # random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.np_original_state)
        # random.setstate(self.random_original_state)


class WeightDistributionFunction:
    def __init__(self, distribution='uniform', **kwargs):
        self.kwargs = kwargs
        self.distribution = distribution

    class UniformDistribution:
        def __init__(self, kwargs):
            self.low = kwargs.get('low', 0)
            self.high = kwargs.get('high', 1)
            self.clip = kwargs.get('clip', 0)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.low == other.low and self.high == other.high
            return False

        def __repr__(self):
            return f"UniformDistribution(low={self.low}, high={self.high})"

        def __call__(self):
            samples = np.random.uniform() * (self.high - self.low) + self.low

            mask = (samples < -self.clip) | (samples > self.clip)  # 超出范围的位置

            for _ in range(100):
                if not np.any(mask):
                    break
                # 对超出范围的位置重新采样
                samples[mask] = np.random.uniform(np.sum(mask)) * (self.high - self.low) + self.low
                mask = (samples < -self.clip) | (samples > self.clip)
            else:
                # 达到最大迭代次数仍未全部满足，最后再 clip 一下（可选）
                logging.warning(f"Warning: Rejection sampling did not fully converge after 100 iterations!")
                samples = np.clip(samples, -self.clip, self.clip)

            return samples

    class NormalDistribution:
        def __init__(self, kwargs):
            self.mu = kwargs.get('mu', 0)
            self.sigma = kwargs.get('sigma', 1)
            self.clip = kwargs.get('clip', 0)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.mu == other.mu and self.sigma == other.sigma
            return False

        def __repr__(self):
            return f"NormalDistribution(loc={self.mu}, scale={self.sigma})"

        def __call__(self):
            samples = np.random.normal(self.mu, self.sigma)
            mask = (samples < -self.clip) | (samples > self.clip)  # 超出范围的位置

            for _ in range(100):
                if mask:
                    break
                # 对超出范围的位置重新采样
                samples = np.random.normal(self.mu, self.sigma)
                mask = (samples < -self.clip) | (samples > self.clip)
            else:
                # 达到最大迭代次数仍未全部满足，最后再 clip 一下（可选）
                logging.warning(f"Warning: Rejection sampling did not fully converge after 100 iterations!")
                samples = np.clip(samples, -self.clip, self.clip)

            return samples

    def __call__(self, *args, **kwargs):
        if self.distribution == 'uniform':
            return self.UniformDistribution(self.kwargs)
        elif self.distribution == 'normal':
            return self.NormalDistribution(self.kwargs)
        else:
            raise ValueError('Invalid distribution!')
