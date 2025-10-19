from copy import deepcopy
import pandas as pd
from src.simulation import Simulation
from joblib import Parallel, delayed
from utils.tools import format_time, format_data, metric_calculation, find_and_delete_invalid_json_cache, create_folder
from stat_test.test_chooser import stat_test_chooser
import os, codecs, shutil, hashlib, joblib, tempfile, json, logging, pickle
from tqdm import tqdm
from algo.algo_chooser import algorithm_chooser


class Evaluation(Simulation):
    def __init__(self, args, configs):
        super().__init__(args, configs)
        self.test_config = configs.test
        self.eval_config = configs.evaluation
        self.cache_config = configs.cache
        self.raw_eval_res = []

    def evaluate_grid_parameter(self, param):
        param, _, _ = self._evaluate_grid_parameter(param)
        metrics_dict = metric_calculation(param['esti_graph'], param['graph'])
        parma_dc = deepcopy(param)
        del parma_dc['data'], parma_dc['graph']
        parma_dc.update(metrics_dict)
        return parma_dc

    def _evaluate_grid_parameter(self, param):
        data_hash = hashlib.md5(str(param['data']).encode('utf-8')).hexdigest()
        cache_path = os.path.join(self.storage.cache_dir, data_hash + '.json')
        cache_config = self.cache_config
        cache_config.cache_path = cache_path
        self.configs.test.true_dag = param['graph']

        algo = algorithm_chooser(self.eval_config.algorithm)
        uit, cit = stat_test_chooser(param['data'], self.configs, cache_config)
        estimate_graph = algo(
            param['data'], uit, cit, self.test_config.alpha
        )
        param['esti_graph'] = estimate_graph
        return param, uit, cit

    def _process_eval_res(self):
        create_folder(self.storage.eval_dir)
        raw_eval_res = pd.DataFrame(self.raw_eval_res)
        try:
            raw_eval_res['algo'] = self.eval_config.algorithm.algo
            raw_eval_res['uit'] = self.test_config.uit
            raw_eval_res['cit'] = self.test_config.cit
            raw_eval_res['alpha'] = self.test_config.alpha
        except AttributeError:
            logging.warning('Key words lost in  evaluation result file !')

        metric_col = ['SHD', 'SHD Anti', 'Normalized SHD', 'Normalized SHD Anti', 'TPR', 'FPR', 'TP', 'FP', 'FN', 'TN',
                      'precision', 'recall', 'F1', 'Accuracy', 'time_spent']
        col2drop = ['g_id', 'd_id', 'seed', 'esti_graph', 'pure_signal']
        col2group = [c for c in raw_eval_res.columns if c not in (col2drop + metric_col)]
        raw_eval_res.drop(col2drop, axis=1, inplace=True)
        res_grouped = raw_eval_res.groupby(col2group)
        data = []
        for n, g in res_grouped:
            # data.append(list(n) + [str(round(g[metric].mean(), 3)) + '(' + str(
            #     round(g[metric].std(), 3)) + ')' if not 'time' in metric else format_time(g[metric].mean()) for metric
            #                        in metric_col])
            data.append(list(n) + [str(round(g[metric].mean(), 3)) + '(' + str(
                round(g[metric].std(), 3)) + ')' for metric in metric_col])
        col2group.extend(metric_col)

        self.eval_res = pd.DataFrame(data=data, columns=col2group)
        self.eval_res.to_excel(os.path.join(self.storage.eval_dir, 'eval_res.xlsx'), index=False)
        raw_eval_res.to_excel(os.path.join(self.storage.eval_dir, 'raw_eval_res.xlsx'), index=False)

    def _evaluation(self):
        json_file_name = [hashlib.md5(str(item['data']).encode('utf-8')).hexdigest() + '.json' for item in
                          self.config_data_graph_list]
        find_and_delete_invalid_json_cache(self.storage.cache_dir, json_file_name)
        loop = tqdm(self.config_data_graph_list, desc='Evaluation of data and graphs:', leave=True)

        if not self.mode.parallel:
            for param in loop:
                res = self.evaluate_grid_parameter(param)
                self.raw_eval_res.append(res)
        else:
            temp_folder = tempfile.mkdtemp(prefix='eval_', dir=self.storage.temp_dir)
            # 创建外层进度条
            try:
                n_jobs = min(len(loop), self.mode.n_jobs)
                with joblib.parallel_backend('loky'):
                    self.raw_eval_res = Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(
                        delayed(self.evaluate_grid_parameter)(param) for param in loop)
            finally:
                shutil.rmtree(temp_folder, ignore_errors=True)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        logging.info('-' * 100)
        logging.info('Evaluation of data and graphs is starting !')
        self._evaluation()
        self._process_eval_res()
        logging.info('Evaluation of data and graphs is over !')
        logging.info('-' * 100)
