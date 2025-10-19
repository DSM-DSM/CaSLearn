import os, logging, shutil, tempfile, pickle, joblib, yaml
from argparse import Namespace
from copy import deepcopy
import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from utils.tools import compute_hash, dump_yaml, dict2namespace, namespace2dict, create_folder, \
    WeightDistributionFunction, comp_signal2noise_ratio
from data_gen.grid_data_generator import data_graph_generator
from tqdm import tqdm


class Simulation(object):
    def __init__(self, args, configs):
        self.key2compare = None
        self.file_loaded = False  # 文件是否是从已有文件中读取的
        self.config_data_graph_list = []
        self.simu_config = configs.simulation
        self.mode = configs.mode
        self.storage = configs.storage
        self.args = args
        self.configs = configs
        # __call__ WeightDistributionFunction !
        self.weight_distribution_function = eval('WeightDistributionFunction')(
            configs.simulation.weight_distribution_function, **configs.simulation.weight_kwargs)()

    @staticmethod
    def check_config(previous_config, current_config):
        """
        :param previous_config:
        :param current_config:
        :return:
        """
        # 仅保留需要比较的键
        previous_hash = compute_hash(namespace2dict(previous_config))
        current_hash = compute_hash(namespace2dict(current_config))
        if current_hash == previous_hash:
            return True
        else:
            logging.info(f'previous hash does not match current hash.')
            return False

    def _get_param(self):
        """
        获取simulation所需的所有参数以及数据
        :return:
        """

        pkl_file_path = os.path.join(self.storage.data_dir, 'config_data_graph_list.pkl')
        if os.path.exists(os.path.join(self.storage.data_dir, 'data_config.yml')):
            with open(os.path.join(self.storage.data_dir, 'data_config.yml'), 'r') as f:
                pre_simu_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            pre_simu_config = {}

        if self.config_data_graph_list:
            pass
        # 如果params_list.pkl文件存在且参数一致，则从文件中读取参数
        elif os.path.exists(pkl_file_path):
            # 由于判断参数一致需要保存的hash值，因此需要读取file
            with open(pkl_file_path, 'rb') as f:
                file = pickle.load(f)
                if self.check_config(pre_simu_config, self.simu_config):
                    self.config_data_graph_list = file['config_data_graph_list']
                    logging.info(
                        f'Data and graph have been loaded from {os.path.join(self.storage.data_dir, "config_data_graph_list.pkl")}')
                    self.file_loaded = True
                else:
                    self._generation()
        else:
            self._generation()

    def _generation(self):
        """
        生成参数与数据
        :return:
        """
        # 使用itertools.product生成所有可能的组合
        param_combinations = list(
            product(self.simu_config.noise_type, self.simu_config.root,
                    self.simu_config.causal_mechanism, self.simu_config.n_samples, self.simu_config.n_points,
                    self.simu_config.exp_degree, list(range(self.simu_config.graph_num)),
                    list(range(self.simu_config.replicates))))

        param_df = pd.DataFrame(
            param_combinations,
            columns=["noise", "root", "causal_mechanism", "npoints", "nodes", "expected_degree",
                     'g_id', 'd_id']
        )

        param_df['d_id'] = list(range(len(param_df)))
        param_df['seed'] = param_df['d_id'] + self.args.seed
        params_combination = [{k: v for k, v in m.items()} for m in param_df.to_dict(orient='records')]
        loop = tqdm(params_combination, desc='Data and Graph Simulation:', leave=True)

        if not self.mode.parallel:
            for param in loop:
                res = self._get_grid_simulation(param)
                self.config_data_graph_list.append(res)
        else:
            temp_folder = tempfile.mkdtemp(prefix='simu_', dir=self.storage.temp_dir)
            try:
                n_jobs = min(len(loop), self.mode.n_jobs)
                with joblib.parallel_backend('loky'):
                    res = Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(
                        delayed(self._get_grid_simulation)(param) for param in loop
                    )
                    self.config_data_graph_list = res
            finally:
                shutil.rmtree(temp_folder, ignore_errors=True)

    def _get_grid_simulation(self, param):
        """
        获取单个网格参数和数据
        :param param:
        :return:
        """
        generator_config = dict2namespace(param)
        generator_config.copula_trans = self.simu_config.copula_trans
        generator_config.copula_trans_kwargs = self.simu_config.copula_trans_kwargs
        generator_config.dag_type = self.simu_config.dag_type
        generator_config.noise_coeff = self.simu_config.noise_coeff
        generator_config.designate = self.simu_config.designate
        generator_config.weight_distribution_function = self.weight_distribution_function
        generator_config = Namespace(**vars(generator_config), **vars(self.simu_config.mechanism_kwargs))
        param_dp = deepcopy(param)
        param_dp['data'], param_dp['pure_signal'], param_dp['graph'] = data_graph_generator(generator_config)
        return param_dp

    def signal2noise_ratio(self):
        count = len(self.config_data_graph_list) / self.simu_config.graph_num / self.simu_config.replicates
        step = self.simu_config.replicates
        model_kwargs_list = ['noise', 'root', 'causal_mechanism', 'npoints', 'nodes',
                             'expected_degree', 'g_id']
        signal2noise_ratio_list = []

        for c in range(int(count)):
            start, end = c * step, (c + 1) * step
            simu = self.config_data_graph_list[start:end]
            signal_stack = np.vstack([s['pure_signal'] for s in simu])
            data_stack = np.vstack([s['data'] for s in simu])

            ratio_tr, sig_tr, noise_tr = comp_signal2noise_ratio(signal_stack, data_stack, metric='tr')
            ratio_eig, sig_eig, noise_eig = comp_signal2noise_ratio(signal_stack, data_stack, metric='eigenvalue')

            item = {k: v for k, v in simu[0].items() if k in model_kwargs_list}
            item.update({
                'sig_tr': sig_tr,
                'noise_tr': noise_tr,
                'ratio_tr': ratio_tr,
                'sig_eig': sig_eig,
                'noise_eig': noise_eig,
                'ratio_eig': ratio_eig,
            })
            signal2noise_ratio_list.append(item)

        df_signal2noise_ratio = pd.DataFrame(
            signal2noise_ratio_list,
            columns=signal2noise_ratio_list[0].keys()
        )
        df_signal2noise_ratio.to_excel(os.path.join(self.storage.data_dir, 'signal2noise_ratio.xlsx'), index=False)

        return df_signal2noise_ratio

    def __call__(self, *args, **kwargs):
        logging.info('Start generating data and graphs !')
        create_folder(self.storage.data_dir)
        self._get_param()

        # self.signal2noise_ratio()

        if not self.file_loaded:
            with open(os.path.join(self.storage.data_dir, 'config_data_graph_list.pkl'), 'wb') as f:
                pickle.dump({'config_data_graph_list': self.config_data_graph_list}, f)
            dump_yaml(os.path.join(self.storage.data_dir, 'data_config.yml'), self.simu_config)
        logging.info('Generation of data and graphs is over !')
