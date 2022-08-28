from churn.config.experiments import Models, Hyperparameters, Search
from churn.config.model_selection import TrainTest
from category_encoders import TargetEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from typing import Any, List, Dict, Optional
from churn.utils.log import Log

import numpy as np
import pandas as pd
import xgboost as xgb


class Experiment:
    MODELS = {
        'random_forest': Models.RANDOM_FOREST,
        'catboost': Models.CATBOOST,
        'xgboost': Models.XGBOOST,
    }
    PARAMS = {
        'random_forest': Hyperparameters.RANDOM_FOREST,
        'catboost': Hyperparameters.CATBOOST,
        'xgboost': Hyperparameters.XGBOOST,
    }
    SEARCH = {
        'random': Search.RANDOM,
        'grid': Search.GRID,
    }

    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 models: List[Any],
                 search_type: str = 'random',
                 mlflow: bool = False,
                 verbose: bool = True,
                 random_state: int = 0,
                 params: Optional[List[Dict]] = None, ):
        self.target = target
        self.data = {
            'original_data': data,
            'X': data.drop(target, axis=1),
            'y': data[target],
        }
        self.models = [self.MODELS.get(model) if isinstance(model, str)
                       else model for model in models]
        self.model_names = [model if isinstance(model, str)
                            else type(model).__name__ for model in models]
        if params is None:
            self.params = [self.PARAMS.get(model) for model in models]
        else:
            self.params = params
        self.search = self.SEARCH.get(search_type)
        self.mlflow = mlflow
        self.verbose = verbose
        self.random_state = random_state

    def train_dev_test_split(self, train_size: float, dev_test_rate: float):

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            self.data['X'],
            self.data['y'],
            random_state=self.random_state,
            train_size=train_size,
        )
        X_dev, X_test, y_dev, y_test = train_test_split(
            X_tmp,
            y_tmp,
            random_state=self.random_state,
            test_size=dev_test_rate,
        )

        self.data['X_train'] = X_train
        self.data['X_dev'] = X_dev
        self.data['X_test'] = X_test
        self.data['y_train'] = y_train
        self.data['y_dev'] = y_dev
        self.data['y_test'] = y_test

        # if 'xgboost' in self.model_names:
        #     dtrain = xgb.DMatrix(X_train, label=y_train)
        #     ddev = xgb.DMatrix(X_dev, label=y_dev)
        #     dtest = xgb.DMatrix(X_test, label=y_test)
        #     self.data['xgboost_dtrain'] = dtrain
        #     self.data['xgboost_ddev'] = ddev
        #     self.data['xgboost_dtest'] = dtest

    def make_preprocessing_xgboost(self):
        data_dmatrix = xgb.DMatrix(data=self.data['X'],
                                   label=self.data['y'])
        self.data['xgboost_dmatrix'] = data_dmatrix

    def make_preprocessing(self,
                           not_features: List[str],
                           numerical_features: List[str],
                           categorical_features: List[str],
                           normalize: bool = False):

        all_features = self.data['X'].columns #list(set(self.data['X'].columns) - set(not_features))
        transformers = make_column_transformer(
            ('drop', not_features),
            # Filling NaNs with a number out ouf the distribution
            # We can do this since features are either categorical
            # or non-negative
            (SimpleImputer(strategy='constant', fill_value=-1), all_features),
            (TargetEncoder(), categorical_features),

        )
        scaler = make_column_transformer(
            (StandardScaler, all_features),
        ) if normalize else make_column_transformer(
            ('passthrough', all_features)
        )

        pipe = Pipeline(
            [
                ('transformers', transformers),
                ('scaler', scaler),
            ]
        )

        self.data['pipeline'] = pipe

        pipe.fit_transform(self.data['X_train'], self.data['y_train'])

        if 'xgboost' in self.model_names:
            self.make_preprocessing_xgboost()
        if 'catboost' in self.model_names:
            self.make_preprocessing_catboost()
