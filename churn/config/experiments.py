from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

import numpy as np


class Models:
    RANDOM_FOREST = RandomForestClassifier
    CATBOOST = CatBoostClassifier
    XGBOOST = XGBClassifier
    LOGISTIC_REGRESSION = LogisticRegression


class Hyperparameters:
    RANDOM_FOREST = {
        'bootstrap': [True, False],
        'max_depth': [2, 5, 10, 15, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [10, 30, 50, 70, 100],
    }
    CATBOOST = {
        'depth': [4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.02, 0.03, 0.04],
        'iterations': [10, 30, 50, 70, 100],
    }
    XGBOOST = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }
    LOGISTIC_REGRESSION = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 1000, 2500, 5000]
    }


class Search:
    GRID = GridSearchCV
    RANDOM = RandomizedSearchCV
