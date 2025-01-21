"""Evaluation of hyperparameter optimization results."""

import numpy as np

from safe_control_gym.hyperparameters.base_hpo import BaseHPO

class HPOEval(BaseHPO):

    def __init__(self,
                 hpo_config,
                 task_config,
                 algo_config,
                 algo='ilqr',
                 task='stabilization',
                 output_dir='./results',
                 safety_filter=None,
                 sf_config=None):
        '''Hyperparameter optimization evaluation class.

        Args:
            hpo_config (dict): HPO configuration.
            task_config (dict): Task configuration.
            algo_config (dict): Algorithm configuration.
            algo (str): Algorithm name.
            task (str): Task name.
            output_dir (str): Output directory.
            safety_filter (str): Safety filter name.
            sf_config (dict): Safety filter configuration.
        '''
        super(HPOEval, self).__init__(hpo_config, task_config, algo_config, algo, task, output_dir, safety_filter, sf_config, False, False)

        if 'vizier_hps' in self.hpo_config:
            self.vizier_hps = self.hpo_config['vizier_hps']
        if 'optuna_hps' in self.hpo_config:
            self.optuna_hps = self.hpo_config['optuna_hps']

    def hp_evaluation(self):
        '''Evaluate the handtuned/optimized hyperparameters and make the comparison.'''
        
        # evaluate handtuned hyperparameters
        handtuned_hps = self.config_to_param(self.hps_config)
        np.random.seed(self.hpo_config.seed)
        self.evaluate(handtuned_hps)
        handtuned_trajs_data_list = self.trajs_data_list
        handtuned_metrics_list = self.metrics_list
        self.plot_results(handtuned_trajs_data_list, handtuned_metrics_list, self.output_dir, '(handtuned hps)')

        # evaluate vizier hyperparameters
        if 'vizier_hps' in self.hpo_config:
            vizier_hps = self.config_to_param(self.vizier_hps)
            np.random.seed(self.hpo_config.seed)
            self.evaluate(vizier_hps)
            vizier_trajs_data_list = self.trajs_data_list
            vizier_metrics_list = self.metrics_list
            self.plot_results(vizier_trajs_data_list, vizier_metrics_list, self.output_dir, '(vizier hps)')
        
        # evaluate optuna hyperparameters
        if 'optuna_hps' in self.hpo_config:
            optuna_hps = self.config_to_param(self.optuna_hps)
            np.random.seed(self.hpo_config.seed)
            self.evaluate(optuna_hps)
            optuna_trajs_data_list = self.trajs_data_list
            optuna_metrics_list = self.metrics_list
            self.plot_results(optuna_trajs_data_list, optuna_metrics_list, self.output_dir, '(optuna hps)')

        trajs_dict = {
            'handtuned hps': handtuned_trajs_data_list,
            'vizier hps': vizier_trajs_data_list if 'vizier_hps' in self.hpo_config else None,
            'optuna hps': optuna_trajs_data_list if 'optuna_hps' in self.hpo_config else None,
        }
        metrics_dict = {
            'handtuned hps': handtuned_metrics_list,
            'vizier hps': vizier_metrics_list if 'vizier_hps' in self.hpo_config else None,
            'optuna hps': optuna_metrics_list if 'optuna_hps' in self.hpo_config else None,
        }

        # Remove None values
        trajs_dict = {k: v for k, v in trajs_dict.items() if v is not None}
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}

        self.plot_results_grid(trajs_dict, metrics_dict, self.output_dir)

    def setup_problem(self):
        """
        Dummy function.
        """
        raise NotImplementedError

    def warm_start(self, params):
        """
        Dummy function.
        """
        raise NotImplementedError
    
    def resume_trials(self):
        """
        Dummy function.
        """
        raise NotImplementedError

    def hyperparameter_optimization(self):
        """
        Dummy function.
        """
        raise NotImplementedError

    def checkpoint(self):
        """
        Dummy function.
        """
        raise NotImplementedError