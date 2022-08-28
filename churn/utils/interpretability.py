"""
Helping you interpret your trained model
"""

from explainerdashboard import RegressionExplainer, ClassifierExplainer, \
    ExplainerDashboard, ExplainerHub
from typing import Any, Dict, List, NoReturn, Optional, Union

import logging
import numpy as np
import os
import pandas as pd
import pickle
import warnings


class Interpretable:
    """
    Interpreting models ran in a Valohai execution.
    You need to download their outputs and save each of them
    in a different directory.
    --------------------------------------------------------
    How to use:
    >>> inter = Interpretable()
    >>> exec_paths = [
    ...    os.path.join(REPO_PATH, 'path', 'to', 'executions', f'exec_{n_exec}')
    ...    for n_exec in EXECS  # iterable with the relevant execution numbers
    ...]
    >>> inter.run_explainer_hub(
    ... root_dir=REPO_PATH,
    ... to_html=False,
    ... to_zip=False,
    ... problem='classification',
    ... exec_paths=exec_paths,
    ... yaml=None,
    ...)

    Check the methods' documentation for more details.
    """
    EXPLAINER = {
        'classification': ClassifierExplainer,
        'regression': RegressionExplainer,
    }

    def load_execution(self, exec_path: str) -> Dict[str, Any]:
        """
        Load data from an experiment execution.
        Keywords available:
            *  'original_data': data loaded from query
            *  'train_idx': array of train indices
            *  'test_idx': array of test indices
            *  'df_train': dataframe of train data
            *  'df_test': dataframe of test data
            *  'X_train': dataframe of train features as used by the model
            *  'X_test': dataframe of test features as used by the model
            *  'y_train': array of train target
            *  'y_test': array of test target
            *  'y_pred': array of test predictions
            *  'baseline': percentage of positive targets in the train set
                           (for binary classification)
            *  'pipeline': pipeline that processes df_{train, test}
            *  'model': estimator used at the end of the pipeline
            *  'explainer': explainer object used to create an ExplainerDashboard

        :param exec_path: directory where the execution outputs are saved
        :return: dictionary with all loaded data, including the model explainer
        """
        # TODO use the data relevant to your case

        original_df = self._access_data(
            f'{exec_path}/dataset.csv')
        train_idx = self._access_data(f'{exec_path}/train_idx.npy')
        test_idx = self._access_data(f'{exec_path}/test_idx.npy')
        df_train = self._access_data(f'{exec_path}/df_train.parquet')
        df_test = self._access_data(f'{exec_path}/df_test.parquet')
        X_train = self._access_data(f'{exec_path}/X_train.parquet')
        X_test = self._access_data(f'{exec_path}/X_test.parquet')

        y_train = self._access_data(f'{exec_path}/y_train_pycaret.npy')
        y_test = self._access_data(f'{exec_path}/y_test_pycaret.npy')

        y_pred = self._access_data(f'{exec_path}/y_pred.npy')
        try:
            baseline = y_train.sum() / y_train.shape[0]
        except Exception as e:
            logging.exception(e)
            baseline = None
        try:
            pipeline = self._access_data(f'{exec_path}/pipeline.pkl')
            model = pipeline._final_estimator
        except Exception as e:
            logging.exception(e)
            model = self._access_data(f'{exec_path}/model.pkl')
            pipeline = None

        explainer = self.explain(exec_path, model, X_test, y_test)

        return {
            'original_data': original_df,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'df_train': df_train,
            'df_test': df_test,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'baseline': baseline,
            'pipeline': pipeline,
            'model': model,
            'explainer': explainer,
        }

    def explain(self,
                exec_path: str,
                model: Any,
                X_test: pd.DataFrame,
                y_test: Optional[pd.Series] = None,
                problem: str = 'classification') -> Union[ClassifierExplainer,
                                                          RegressionExplainer]:
        """
        Creates the objected needed to create an ExplainerDashboard object later.
        :param exec_path: : directory where the execution outputs are saved.
        :param model: the estimator object.
        :param X_test: Test dataframe
        :param y_test: Test target series
        :param problem: either 'classification' or 'regression'
        :return: the explainer object to be used as argument for ExplainerDashboard
        """
        filepath = f'{exec_path}/explainer.dill'
        try:
            explainer = self.EXPLAINER[problem].from_file(filepath)
        except Exception as e:
            logging.exception(e)
            warnings.warn(f'Creating explainer and saving it in {filepath}')
            explainer = self.EXPLAINER[problem](model, X_test, y_test)
            explainer.dump(filepath)
        return explainer

    def create_dashboards(self,
                          exec_paths: List[str],
                          **kwargs) -> List[ExplainerDashboard]:
        """
        :param exec_paths: list of paths to all executions.
        :param kwargs: ExplainerDashboard parameters
        :return: list containing dashboard objects
        """
        dashboards = []
        for path in exec_paths:
            title = path.split('/')[-1]
            yaml = f'{title}_dashboard.yaml'
            try:
                dash = ExplainerDashboard.from_config(yaml)
            except TypeError:
                warnings.warn(f'Creating dashboard and saving it in {yaml}')
                data = self.load_execution(exec_path=path)
                dash = ExplainerDashboard(data['explainer'],
                                          name=title, title=title, **kwargs)
                dash.to_yaml(yaml)
            dashboards.append(dash)

        return dashboards

    def initialize_hub(self,
                       root_dir: str,
                       to_html: bool = False,
                       to_zip: bool = False,
                       problem: str = 'classification',
                       exec_paths: Optional[List[str]] = None,
                       yaml: Optional[str] = None,
                       **kwargs) -> ExplainerHub:
        """
        :param root_dir: the root directory storing the configuration files
        :param to_html: whether to save the hub as html
        :param to_zip: whether to save the hub as zip
        :param problem: 'classification' or 'regression'
        :param exec_paths: list of paths of all the execution to interpret
        :param yaml: if hub's is YAML available, provide its path
        :param kwargs: ExplainerDashboard parameters
        :return: explainer hub object
        """

        try:
            hub = ExplainerHub.from_config(yaml)
        except TypeError:
            warnings.warn(f'YAML file does not exists, '
                          f'creating hub and saving it in {yaml}')
            if yaml is None:
                yaml = os.path.join(root_dir, 'hub.yaml')
            dashboards = self.create_dashboards(exec_paths, problem, **kwargs)
            hub = ExplainerHub(dashboards)
            hub.to_yaml(yaml)

        if to_html:
            hub.to_html()
        if to_zip:
            hub.to_zip(yaml.replace('yaml', 'zip'))

        return hub

    def run_explainer_hub(self,
                          root_dir: str,
                          to_html: bool = False,
                          to_zip: bool = False,
                          problem: str = 'classification',
                          exec_paths: Optional[List[str]] = None,
                          yaml: Optional[str] = None,
                          **kwargs) -> NoReturn:
        """
        Runs the entire pipeline from creating an explainer to running a
        hub with multiple dashboards.

        :param root_dir: the root directory storing the configuration files
        :param to_html: whether to save the hub as html
        :param to_zip: whether to save the hub as zip
        :param problem: 'classification' or 'regression'
        :param exec_paths: list of paths of all the execution to interpret
        :param yaml: if hub's is YAML available, provide its path
        :param kwargs: parameters for explainer, dashboard, hub
        """

        hub = self.initialize_hub(
            root_dir, to_html, to_zip, problem, exec_paths, yaml, **kwargs
        )
        hub.run(**kwargs)

    @staticmethod
    def _access_data(path: str) -> Any:
        try:
            if 'parquet' in path:
                return pd.read_parquet(path)
            if 'csv' in path:
                return pd.read_csv(path)
            elif 'npy' in path:
                return np.load(path)
            elif 'pkl' in path:
                with open(path, 'rb') as file:
                    return pickle.load(file)
        except Exception as e:
            logging.exception(e)
            warnings.warn('The file given is not parquet, csv, pickle'
                          ' or numpy. Returning None')
            return None
