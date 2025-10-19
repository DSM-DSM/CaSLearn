import logging
import time
import networkx as nx
from collections import Counter
from utils.tools import dict2namespace


def _vea(data, uit, cit, alpha, threshold, algo_config):
    """
    :param data:
    :param alpha:
    :param threshold:
    :param kwargs:
    :return:
    """
    from algo.algo_chooser import algorithm_chooser

    start = time.time()
    pc_fit_list, edge_count = [], []
    total_candidate_num = len(uit) * len(cit)

    for ut_method, ut_ins in uit.items():
        for ct_method, ct_ins in cit.items():
            algo = algorithm_chooser(algo_config)
            pc_fit = algo(data, ut_ins, ct_ins, alpha=alpha)
            pc_fit.to_nx_graph()
            pc_fit_list.append(pc_fit)
            edges = list(pc_fit.nx_graph.edges)  # CPDAG
            edge_count.extend(edges)
    edge_count_counter = Counter(edge_count)
    edges_remain = {key: count / total_candidate_num for key, count in edge_count_counter.items() if
                    count / total_candidate_num >= threshold}
    pc_fit.nx_graph = nx.DiGraph(edges_remain.keys())
    pc_fit.nx_skel = pc_fit.nx_graph.to_undirected()
    pc_fit.PC_elapsed = time.time() - start
    return pc_fit


def vea(data, uit, cit, alpha, threshold, **kwargs):
    kwargs['algo'] = 'pc'
    config = dict2namespace(kwargs)
    return _vea(data, uit, cit, alpha, threshold, config)


def deduce_dep_voting_edge(data, uit, cit, alpha, threshold, **kwargs):
    kwargs['algo'] = 'deduce_dep_pc'
    config = dict2namespace(kwargs)
    return _vea(data, uit, cit, alpha, threshold, config)

def vea_plus(data, uit, cit, alpha, threshold, **kwargs):
    
    return _vea(data, uit, cit, alpha, threshold, **kwargs)