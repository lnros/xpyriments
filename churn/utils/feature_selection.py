import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance


def Boruta_feature_selection(X_train, y_train, X_test, random_state=0, alpha=0.05):
    X_train_labeld = X_train[y_train != -1]
    y_train_labeld = y_train[y_train != -1]
    X_train_numeric = X_train_labeld.select_dtypes(include=np.number)
    # define Boruta feature selection method
    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=random_state, alpha=alpha)

    # find all relevant features
    feat_selector.fit(X_train_numeric.values, y_train_labeld)
    cols_to_drop = [col for ind, col in enumerate(X_train_numeric.columns) if
                    not ((feat_selector.support_[ind]) | (feat_selector.support_weak_[ind]))]
    X_train_selected, X_test_selected = X_train.drop(cols_to_drop, axis=1), X_test.drop(cols_to_drop, axis=1)
    return X_train_selected, X_test_selected


def drop_by_correlation(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return cols_to_drop


def feature_selection_by_permutation_importance(model, X_train, y_train, n_folds=5, random_state=0,
                                                n_permutations=5, threshold=0.0):
    per_importance = pd.Series(np.repeat(0, X_train.shape[1]), index=X_train.columns)
    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=1, random_state=random_state)
    for train, test in cv.split(X_train, y_train):
        train_inds, test_inds = X_train.index[train], X_train.index[test]
        x_train, x_val = X_train.loc[train_inds, :], X_train.loc[test_inds, :]
        y_tr, y_val = y_train.loc[train_inds], y_train.loc[test_inds]
        model.fit(x_train, y_tr)
        res = permutation_importance(model, x_val, y_val,  n_repeats=n_permutations,
                                     n_jobs=-1, scoring='average_precision')
        per_importance += res.importances_mean
    per_importance /= n_folds
    return per_importance[per_importance > threshold].index