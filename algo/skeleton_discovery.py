from itertools import combinations
from numpy import ndarray
from typing import List
from tqdm.auto import tqdm
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
import numpy as np
from algo.graphs import CausalGraphPlus


def deduce_dep(X, Y, Z, k, alpha, sepsets, consets=None, ci_tester=None):
    '''
    recursively deduce a dependence statement from CIT results
    Args:
        X: variable of interest from CI query (X;Y|Z)
        Y: variable of interest from CI query (X;Y|Z)
        Z: conditioning set from CI query (X;Y|Z)
        k: recursion threshold for deduce-dep
        alpha: the significance level for CIT to use
        sepsets: a dictionary of CI queries with independence results
        consets: a dictionary of CI queries with dependence results
        ci_tester: CIT to use

    Returns: whether dependence is deducible or not (Boolean)

    '''
    if consets is None:
        consets = dict()
    if len(Z) > k:
        for z in Z:
            remaining_Z = tuple(list((set(Z) - {z})))

            for A, B, C in [(X, Y, remaining_Z), (X, z, remaining_Z), (Y, z, remaining_Z)]:
                is_already_identified = False

                # 判断A ⊥ B | C是否已经判断
                if tuple(sorted([A, B])) in sepsets and sepsets[tuple(sorted([A, B]))] == C:
                    pval = 1
                    is_already_identified = True

                elif tuple(sorted([A, B])) in consets and consets[tuple(sorted([A, B]))] == C:
                    pval = 0

                else:
                    pval = ci_tester(A, B, C)

                # pval > alpha indicates that A and B are independent and should be put in sepsets
                if pval > alpha:
                    if not is_already_identified:
                        # deduce_dep returns false means we have to trust the result of ci_test
                        if not deduce_dep(A, B, C, k, alpha, sepsets, consets, ci_tester=ci_tester):
                            sepsets[tuple(sorted([A, B]))] = C
                        else:
                            consets[tuple(sorted([A, B]))] = C

                else:
                    consets[tuple(sorted([A, B]))] = C

                if tuple(sorted([X, Y])) not in sepsets:
                    if (tuple(sorted([X, z])) in sepsets and sepsets[(tuple(sorted([X, z])))] == remaining_Z) or (
                            tuple(sorted([Y, z])) in sepsets and sepsets[(tuple(sorted([Y, z])))] == remaining_Z):
                        return True

            if tuple(sorted([X, Y])) in sepsets and sepsets[tuple(sorted([X, Y]))] == remaining_Z:
                if (tuple(sorted([X, z])) not in sepsets) and (tuple(sorted([Y, z])) not in sepsets):
                    return True
    return False


def skeleton_discovery(data: ndarray, uit, cit, alpha: float, stable: bool = True,
                       background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False,
                       show_progress: bool = False, node_names: List[str] | None = None, **kwargs) -> CausalGraphPlus:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraphPlus(no_of_var, node_names)
    cg.set_ind_test(uit=uit, cit=cit)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.idep_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg


def skeleton_discovery_deduce_dep(data: ndarray, uit, cit, alpha: float, stable: bool = True,
                                  background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False,
                                  show_progress: bool = False, node_names: List[str] | None = None,
                                  deduce_dep_k: int = 1, **kwargs) -> CausalGraphPlus:
    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraphPlus(no_of_var, node_names)
    cg.set_ind_test(uit=uit, cit=cit)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None

    # 为适应deduce_dep设计的独立和非独立集字典
    connected_sets4deduce_dep = dict()
    seperated_sets4deduce_dep = dict()
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.idep_test(x, y, S)
                    if p > alpha:  # means conditional independent
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            # 倘若不可以通过deductive rule拒绝条件独立性,则维持高维的条件独立性判断
                            if not deduce_dep(x, y, S, deduce_dep_k, alpha, seperated_sets4deduce_dep,
                                              connected_sets4deduce_dep, cg.citest_instance):
                                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                                if edge1 is not None:
                                    cg.G.remove_edge(edge1)
                                edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                                if edge2 is not None:
                                    cg.G.remove_edge(edge2)
                                seperated_sets4deduce_dep[tuple(sorted([x, y]))] = S
                                seperated_sets4deduce_dep[tuple(sorted([y, x]))] = S
                                append_value(cg.sepset, x, y, S)
                                append_value(cg.sepset, y, x, S)
                                break
                            # 若能通过deductive rule拒绝条件独立性，则不必删除 x <--> y这两条edges
                            else:
                                connected_sets4deduce_dep[tuple(sorted([x, y]))] = S
                                connected_sets4deduce_dep[tuple(sorted([y, x]))] = S
                        else:
                            if not deduce_dep(x, y, S, deduce_dep_k, alpha, seperated_sets4deduce_dep,
                                              connected_sets4deduce_dep, cg.citest_instance):
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                            else:
                                connected_sets4deduce_dep[tuple(sorted([x, y]))] = S
                                connected_sets4deduce_dep[tuple(sorted([y, x]))] = S
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                        connected_sets4deduce_dep[tuple(sorted([x, y]))] = S
                        connected_sets4deduce_dep[tuple(sorted([y, x]))] = S
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg


def vpca_skeleton_discovery(uit, cit, alpha, no_of_var, stable: bool = True,
                            background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False,
                            show_progress: bool = False, node_names: List[str] | None = None, **kwargs):
    assert 0 < alpha < 1

    cg = CausalGraphPlus(no_of_var, node_names)
    cg.set_ind_test(uit=uit, cit=cit)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.vpca_idep_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg


def vpca_skeleton_discovery_deduce_dep(uit, cit, alpha, no_of_var, deduce_dep_k: int = 1, stable: bool = True,
                                       background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False,
                                       show_progress: bool = False, node_names: List[str] | None = None, **kwargs):
    assert 0 < alpha < 1

    cg = CausalGraphPlus(no_of_var, node_names)
    cg.set_ind_test(uit=uit, cit=cit)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None

    # 为适应deduce_dep设计的独立和非独立集字典
    connected_sets4deduce_dep = dict()
    seperated_sets4deduce_dep = dict()
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.vpca_idep_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            if not deduce_dep(x, y, S, deduce_dep_k, alpha, seperated_sets4deduce_dep,
                                              connected_sets4deduce_dep, cg.citest_instance):
                                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                                if edge1 is not None:
                                    cg.G.remove_edge(edge1)
                                edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                                if edge2 is not None:
                                    cg.G.remove_edge(edge2)
                                seperated_sets4deduce_dep[tuple(sorted([x, y]))] = S
                                seperated_sets4deduce_dep[tuple(sorted([y, x]))] = S
                                append_value(cg.sepset, x, y, S)
                                append_value(cg.sepset, y, x, S)
                                break
                            # 若能通过deductive rule拒绝条件独立性，则不必删除 x <--> y这两条edges
                            else:
                                connected_sets4deduce_dep[tuple(sorted([x, y]))] = S
                                connected_sets4deduce_dep[tuple(sorted([y, x]))] = S
                        else:
                            if not deduce_dep(x, y, S, deduce_dep_k, alpha, seperated_sets4deduce_dep,
                                              connected_sets4deduce_dep, cg.citest_instance):
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                            else:
                                connected_sets4deduce_dep[tuple(sorted([x, y]))] = S
                                connected_sets4deduce_dep[tuple(sorted([y, x]))] = S
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                        connected_sets4deduce_dep[tuple(sorted([x, y]))] = S
                        connected_sets4deduce_dep[tuple(sorted([y, x]))] = S
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg
