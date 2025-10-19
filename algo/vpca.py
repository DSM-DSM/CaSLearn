import logging

import numpy as np
from copy import deepcopy
from causallearn.graph.GraphClass import CausalGraph
from algo.skeleton_discovery import vpca_skeleton_discovery, vpca_skeleton_discovery_deduce_dep
from algo.skeleton_orientation import orientation
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
import time


def vpca(uit, cit, alpha, p, deduce_dep: bool = False, stable: bool = True,
         background_knowledge: BackgroundKnowledge | None = None, verbose: bool = False, show_progress: bool = False,
         node_names=None, uc_rule=0, uc_priority=2, **kwargs):
    start = time.time()
    if not deduce_dep:
        cg_1 = vpca_skeleton_discovery(uit, cit, alpha, p, stable, background_knowledge=background_knowledge,
                                       verbose=verbose, show_progress=show_progress, node_names=node_names, **kwargs)
    else:
        try:
            k = kwargs['k']
        except KeyError:
            logging.warning('Key Parameter k of Deduce Dependence is Not specified, set k to 1 !')
            k = 1
        cg_1 = vpca_skeleton_discovery_deduce_dep(uit, cit, alpha, p, k, stable,
                                                  background_knowledge=background_knowledge,
                                                  verbose=verbose, show_progress=show_progress, node_names=node_names,
                                                  **kwargs)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    cg = orientation(cg_1, alpha, uc_rule, uc_priority, background_knowledge)
    end = time.time()

    cg.PC_elapsed = end - start
    cg.to_nx_skeleton()
    cg.to_nx_graph()
    return cg
