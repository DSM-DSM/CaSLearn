import logging
from stat_test.test_chooser import stat_test_chooser


def algorithm_chooser(algo_config):
    if not hasattr(algo_config, 'algo'):
        logging.error("Invalid eval_config: missing 'algo' attribute")

    def algo_wrapper(algo_func, **algo_kwargs):
        """
        Unify the parameters of an algorithm with multiple different parameters into four
        """

        def func_wrapper(data, uit, cit, alpha):
            return algo_func(data, uit, cit, alpha, **algo_kwargs)

        return func_wrapper

    algo_kwargs = algo_config.__dict__

    if algo_config.algo == 'pc':
        from algo.pc import pc
        return algo_wrapper(pc, **algo_kwargs)

    elif algo_config.algo == 'deduce_dep_pc':
        from algo.pc import deduce_dep_pc
        return algo_wrapper(deduce_dep_pc, **algo_kwargs)

    elif algo_config.algo == 'copula_pc':
        if algo_config.copula_pc_kwargs.copula_pc_stable:
            from algo.copula_pc import copula_pc_stable
            return algo_wrapper(copula_pc_stable, **algo_kwargs)
        else:
            from algo.copula_pc import copula_pc
            return algo_wrapper(copula_pc, **algo_kwargs)

    elif algo_config.algo == 'voting_edge':
        from algo.vea import vea
        return algo_wrapper(vea, **algo_kwargs)

    elif algo_config.algo == 'deduce_dep_voting_edge':
        from algo.vea import deduce_dep_voting_edge
        return algo_wrapper(deduce_dep_voting_edge, **algo_kwargs)

    elif algo_config.algo == 'vea_plus':
        from algo.vea import vea_plus
        return algo_wrapper(vea_plus, **algo_kwargs)
    else:
        logging.error(f'Algorithm {algo_config.algo} does not supported !')
        raise ValueError
