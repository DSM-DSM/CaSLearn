from stat_test.init import *
from argparse import Namespace


def stat_test_chooser(data, config, cache_config):
    algo = config.evaluation.algorithm.algo
    test_config = Namespace(**vars(config.test), **vars(cache_config))
    if algo in ['pc', 'deduce_dep_pc']:
        uit = _stat_test_chooser(data, test_config, config.test.uit)
        cit = _stat_test_chooser(data, test_config, config.test.cit)
    elif algo in ['voting_edge', 'deduce_dep_voting_edge']:
        uit = {method: _stat_test_chooser(data, test_config, method) for method in VE_UIT_METHODS}
        cit = {method: _stat_test_chooser(data, test_config, method) for method in VE_CIT_METHODS}
    elif algo == 'copula_pc':
        from algo.copula_pc import gbs_sampling
        import time
        test_config.t_start = time.time()
        if config.evaluation.algorithm.copula_pc_kwargs.copula_pc_stable:
            cor_matrix_list, effective_number = gbs_sampling(data, config)
            test_config.effective_number = effective_number
            uit, cit = [], []
            for cor_matrix in cor_matrix_list:
                test_config.cor_matrix = cor_matrix
                uit.append(_stat_test_chooser(data, test_config, copc))
                cit.append(_stat_test_chooser(data, test_config, copc))
        else:
            cor_matrix, effective_number = gbs_sampling(data, config)
            test_config.cor_matrix, test_config.effective_number = cor_matrix, effective_number
            uit = _stat_test_chooser(data, test_config, copc)
            cit = _stat_test_chooser(data, test_config, copc)
    else:
        logging.warning(f'Algorithm does {algo} not support yet !')
        raise ValueError("Unknown algorithm: {}".format(algo))
    return uit, cit


def _stat_test_chooser(data, test_config, test_name):
    test_kwargs = test_config.__dict__
    if test_name == fisherz:
        from stat_test.u_cit.pcor import Fisherz
        return Fisherz(data, **test_kwargs)
    elif test_name == spearman:
        from stat_test.u_cit.pcor import Spearmanz
        return Spearmanz(data, **test_kwargs)
    elif test_name == kendall:
        from stat_test.u_cit.pcor import Kendallz
        return Kendallz(data, **test_kwargs)
    elif test_name == robustQn:
        from stat_test.u_cit.pcor import RobustQnz
        return RobustQnz(data, **test_kwargs)
    elif test_name == kci:
        from stat_test.u_cit.kernel import KCI
        return KCI(data, **test_kwargs)
    # elif test_name in [chisq, gsq]:
    #     return Chisq_or_Gsq(data, test_name_name=test_name, **kwargs)
    # elif test_name == mv_fisherz:
    #     return MV_FisherZ(data, **kwargs)
    # elif test_name == mc_fisherz:
    #     return MC_FisherZ(data, **kwargs)
    elif test_name == d_separation:
        # true dag is required!
        from stat_test.ground_truth import D_Separation
        assert hasattr(test_config, 'true_dag'), "D-separation requires true dag !"
        assert isinstance(test_config.true_dag, nx.DiGraph), "True dag must be a networkx.DiGraph"
        return D_Separation(data, **test_kwargs)
    elif test_name == hsic:
        from stat_test.u_cit.hsic import HSIC
        return HSIC(data, **test_kwargs)
    elif test_name == classifier:
        from stat_test.u_cit.classfier import Classifier
        return Classifier(data, **test_kwargs)
    elif test_name == conditional_distance:
        from stat_test.u_cit.cond_dist import ConditionalDistance
        return ConditionalDistance(data, **test_kwargs)
    elif test_name == gcm:
        from stat_test.u_cit.gcm import GCM
        return GCM(data, **test_kwargs)
    elif test_name == knn:
        from stat_test.cit.knn.knn import KNN
        return KNN(data, **test_kwargs)
    elif test_name == gan:
        from stat_test.cit.gan.gan import GAN
        return GAN(data, **test_kwargs)
    elif test_name == lp:
        from stat_test.cit.lp.lp import Lp
        return Lp(data, **test_kwargs)
    elif test_name == diffusion:
        from stat_test.cit.diffusion.diffusion import Diffusion
        return Diffusion(data, **test_kwargs)
    elif test_name == rruit:
        from stat_test.uit.rr import RecurrentRateUIT
        return RecurrentRateUIT(data, **test_kwargs)
    elif test_name == copc:
        from stat_test.u_cit.copc import Copula_Fisherz
        return Copula_Fisherz(data, **test_kwargs)
    elif test_name == wgcm:
        from stat_test.cit.wgcm import WeightedGCM
        return WeightedGCM(data, **test_kwargs)
    elif test_name == dgan:
        from stat_test.cit.dgan.dgan import DoubleGAN
        return DoubleGAN(data, **test_kwargs)
    elif test_name == pcor:
        from stat_test.u_cit.voting import PartialCor
        return PartialCor(data, **test_kwargs)
    elif test_name == vt:
        from stat_test.u_cit.voting import Voting_Test
        return Voting_Test(data, **test_kwargs)
    else:
        raise ValueError("Unknown test_name: {}".format(test_name))
