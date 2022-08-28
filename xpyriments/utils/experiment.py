"""
Experiment Utils
"""

from churn.config.constants import Constant as const
from churn.config.features import Features
from churn.config.model_selection import TrainTest
from pycaret.classification import *
from sklearn.metrics import accuracy_score, average_precision_score, \
    confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from churn.utils.log import Log

import json
import logging
import os
import pandas as pd

LOG = Log()
pd.set_option('display.width', 1000)


class Experiment:

    def __init__(self,
                 data: pd.DataFrame,
                 experiment_name: str):

        self.data = data
        self.df_test = None
        self.experiment_name = experiment_name
        self.config_name = f'{self.experiment_name}_cfg.pkl'
        self.current_experiment_dir = self._get_latest_experiment()
        self._full_config_name = f'{self.current_experiment_dir}/' \
                                 f'{self.config_name}' \
            if self.current_experiment_dir is not None else None
        self.best_model = None
        self.model_name = None

    def initialize_train_setup(self,
                               normalize: bool = False,
                               fix_imbalance: bool = False,
                               feature_selection: bool = True,
                               feature_selection_method: str = 'boruta',
                               html=False,
                               log_experiment=True,
                               log_plots=False,
                               log_data=False):
        LOG.start_logging()
        df_train, df_tmp = train_test_split(
            self.data,
            train_size=TrainTest.TRAIN_SIZE,
            random_state=const.RANDOM_SEED,
            stratify=self.data[Features.TARGET],
        )
        df_dev, df_test = train_test_split(
            df_tmp,
            test_size=TrainTest.DEV_TEST_RATE,
            random_state=const.RANDOM_SEED,
            stratify=df_tmp[Features.TARGET],
        )
        # Used for prediction
        self.df_test = df_test
        try:
            setup(
                experiment_name=self.experiment_name,
                data=df_train,
                test_data=df_dev,
                target=Features.TARGET,
                categorical_features=list(Features.CATEGORICAL),
                numeric_features=list(Features.NUMERICAL),
                date_features=list(Features.DATES),
                ignore_features=list(Features.NOT_FEATURES),
                session_id=const.RANDOM_SEED,
                normalize=normalize,
                fix_imbalance=fix_imbalance,
                feature_selection=feature_selection,
                feature_selection_method=feature_selection_method,
                html=html,
                silent=True,
                log_experiment=log_experiment,
                log_plots=log_plots,
                log_data=log_data,
            )
        except Exception as e:
            LOG.logger.error(e)
        else:
            self._create_experiment_dir()
            self._full_config_name = f'{self.current_experiment_dir}/' \
                                     f'{self.config_name}'
            save_config(self._full_config_name)

    def train_model(self, sort='F1'):
        LOG.start_logging()
        load_config(self._full_config_name)
        self.best_model = compare_models(
            include=['rf', 'lightgbm', 'xgboost', 'catboost', 'ada'],
            sort=sort
        )
        self.model_name = type(self.best_model).__name__

        save_model(self.best_model,
                   f'{self.current_experiment_dir}/{self.model_name}')
        y_test = get_config('y_test')
        X_test = get_config('X_test')
        y_pred = self.best_model.predict(X_test)
        y_scores = self.best_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_scores)
        matrix = confusion_matrix(y_test, y_pred)
        result = matrix.tolist()
        print(json.dumps({'cm_data': result}))
        generalization_results = pd.DataFrame(
            {
                'model_name': [self.model_name],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1': [f1],
                'AP': [ap],
            }
        )
        # logging metrics
        for col in generalization_results.columns:
            try:
                metric = generalization_results[col].values[0]
            except Exception as e:
                logging.exception(e)
            LOG.logger.debug(f'{col}: {metric}')

    def analyze(self,
                dash_kwargs: Optional[Dict] = None,
                run_kwargs: Optional[Dict] = None):
        if dash_kwargs is None:
            dash_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}
        dashboard(
            estimator=self.best_model,
            display_format='external',
            dashboard_kwargs=dash_kwargs,
            run_kwargs=run_kwargs
        )

    def predict(self, data):
        load_config(self.config_name)
        pipeline = load_model(self.model_name)
        model = pipeline.named_steps.trained_model
        try:
            predict_df = predict_model(model, data=data, raw_score=True)
        except Exception as e:
            logging.exception(e)
        else:

            LOG.logger.debug(f'prediction_shape: {predict_df.shape}')
            LOG.logger.debug(
                'prediction_distribution: '
                f"{predict_df['Label'].value_counts(normalize=True).to_json()}"
            )
        filename = 'predict_output.parquet'
        predict_df.to_parquet(f'{self.current_experiment_dir}/{filename}')

    def _create_experiment_dir(self):
        if not os.path.isdir(const.EXPERIMENTS_DIR):
            os.mkdir(const.EXPERIMENTS_DIR)
        experiment_dir = os.path.join(const.EXPERIMENTS_DIR,
                                      self.experiment_name)
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)
        i = 0
        while True:
            current_experiment = os.path.join(experiment_dir,
                                              self.experiment_name + f'_{i}')
            if not os.path.isdir(current_experiment):
                os.mkdir(current_experiment)
                break
            i += 1
        self.current_experiment_dir = current_experiment

    def _get_latest_experiment(self):
        experiment_dir = f'{const.EXPERIMENTS_DIR}/{self.experiment_name}'
        dirs = [x[0].split('/')[-1].split('_')[-1]
                for x in os.walk(experiment_dir)]
        numeric_dirs = [int(dir_) if dir_.isnumeric() else -1 for dir_ in dirs]
        try:
            latest_exp = max(numeric_dirs)
        except ValueError:
            # case when first experiment
            latest_exp = -1
        if latest_exp == -1:
            latest_dir = None
        else:
            latest_dir = os.path.join(experiment_dir,
                                      self.experiment_name + f'_{latest_exp}')

        return latest_dir
