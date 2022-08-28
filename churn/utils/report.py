"""
Report Utils
"""
from config.constants import Constant as const
from pycaret.classification import *
from tqdm import tqdm
from typing import Any, Optional, AnyStr, Dict, Union, List

import numpy as np
import pandas as pd
import shap


def create_results_df(prediction_output_df: pd.DataFrame) -> pd.DataFrame:
    res_df = prediction_output_df[['account_id', 'account_name',
                                   'opportunity_name', 'territory',
                                   'opportunity_created_date', 'expiry_date',
                                   'relevant_date', 'original_n_servers',
                                   'original_product', 'account_type',
                                   'top_subscription']]
    res_df['rating'] = np.where(
        prediction_output_df['Score_1'] < const.LOW_THRESHOLD,
        'low',  # 'Churn Is Unlikely',
        np.where(
            prediction_output_df['Score_1'] < const.MEDIUM_THRESHOLD,
            'medium',  # 'May Churn: Get In Touch',
            np.where(
                prediction_output_df['Score_1'] < const.HIGH_THRESHOLD,
                'medium',  # 'May Churn: Get In Touch',
                'high',  # 'High Risk of Churn: Act Now'
            )))
    res_df = res_df.reset_index(drop=True)
    return res_df


def create_output_table_features(result_df: pd.DataFrame,
                                 model: Any,
                                 cols_to_drop: Union[List[AnyStr], pd.Index],
                                 X_pred: pd.DataFrame,
                                 n_largest: int = 5,
                                 identifier: AnyStr = 'account_id',
                                 target_population: AnyStr = 'accounts',
                                 X_test_disc: Optional[Any] = None) \
        -> pd.DataFrame:
    """
    :param result_df: a df contains 3 columns: account, class_pred, and prob
    :param model: a decision-tree-based model (e.g., random forest, catboost)
    :param cols_to_drop: which columns
    :param X_pred: the tests set (df)
    :param n_largest: how many features to present
                      (both negative and positive effect)
    :param identifier: result_df object identifier
    :param target_population: name of the target population
    :return: a df in shape (X_test.shape[0]*n_largest*2, 6),
             where each row contains a feature name and value
             that effect (either positive or negative)
             the output of this account. Each account has n_largest*2 rows, and
             class pred and prob are the same along these rows.
    """
    explainer = shap.TreeExplainer(model)
    shap_mat = explainer.shap_values(X_pred)
    shap_df = pd.DataFrame(shap_mat, columns=X_pred.columns)
    if cols_to_drop is not None:
        shap_df = shap_df.drop(cols_to_drop, axis=1)
    full_shap_df = pd.DataFrame(columns=X_pred.columns)
    for col in X_pred.columns:
        if col in shap_df.columns:
            full_shap_df[col] = shap_df[col]
        else:
            full_shap_df[col] = np.zeros(X_pred.shape[0])
    order = np.argsort(-shap_df.values, axis=1)[:, :n_largest]
    order1 = np.argsort(-shap_df[shap_df > 0].values, axis=1)[:, :n_largest]
    largest_positive = pd.DataFrame(shap_df.columns[order],
                                    columns=[str(i + 1) + ' positive' for i in
                                             range(n_largest)],
                                    index=shap_df.index)
    largest_positive[order != order1] = np.nan
    order = np.argsort(shap_df.values, axis=1)[:, :n_largest]
    order1 = np.argsort(shap_df[shap_df < 0].values, axis=1)[:, :n_largest]
    largest_negative = pd.DataFrame(shap_df.columns[order],
                                    columns=[str(i + 1) + ' negative' for i in
                                             range(n_largest)],
                                    index=shap_df.index)
    largest_negative[order != order1] = np.nan
    result = pd.concat([largest_positive, largest_negative], axis=1)
    X_test_values = X_test_disc if X_test_disc is not None else X_pred
    quantiles_lower = {col: np.quantile(X_test_values[col], 0.33) for col in
                       X_test_values.columns if
                       X_test_values.dtypes[col] in (
                           'int', 'float', 'float32')}
    quantiles_upper = {col: np.quantile(X_test_values[col], 0.66) for col in
                       X_test_values.columns if
                       X_test_values.dtypes[col] in (
                           'int', 'float', 'float32')}

    shap_lists = pd.Series(result.values.tolist())
    mask_array = pd.get_dummies(shap_lists.apply(pd.Series).stack()).sum(
        level=0)
    df_mask = pd.DataFrame(columns=X_pred.columns)
    for col in X_pred.columns:
        if col in mask_array.columns:
            df_mask[col] = mask_array[col]
    df_mask = df_mask[X_pred.columns]
    df_filtered = X_test_values.where(df_mask == 1, other=np.nan)
    for col in df_filtered.columns:
        if df_filtered.dtypes[col] in ('int', 'float', 'float32'):
            df_filtered[col] = df_filtered[col].apply(
                lambda x:
                _grade_feature(x, target_population,
                               quantiles_lower, quantiles_upper, col)
            )

    output_with_features = pd.concat([df_filtered, result_df], axis=1)
    output_with_features = pd.melt(output_with_features,
                                   id_vars=result_df.columns).dropna()
    output_with_shap = full_shap_df.where(df_mask == 1, np.nan)
    output_with_shap[identifier] = result_df[identifier]
    output_with_shap = pd.melt(output_with_shap, id_vars=[identifier]).dropna()
    output_df = output_with_features.merge(output_with_shap,
                                           on=[identifier, 'variable'])
    output_df.columns = ['account_id', 'rating', 'prob',
                         'feature', 'feature_value', 'shap_importance']

    return output_df


def get_level_of_effect(output_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a column of level of effect to the input Dataframe"""

    output_df['level_of_effect'] = ''
    for i in tqdm(range(output_df.shape[0])):
        output_df_cur_feature = output_df[
            output_df.feature == output_df.loc[i, 'feature']]
        if output_df.loc[i, 'shap_importance'] > 0:
            median_positive = np.median(
                output_df_cur_feature.loc[
                    output_df_cur_feature['shap_importance'] > 0,
                    'shap_importance'])
            if output_df.loc[i, 'shap_importance'] > median_positive:
                output_df.at[i, 'level_of_effect'] = 'strong'
            else:
                output_df.at[i, 'level_of_effect'] = 'weak'
        else:
            median_negative = np.median(
                output_df_cur_feature.loc[
                    output_df_cur_feature['shap_importance'] <= 0,
                    'shap_importance'])
            if output_df.loc[i, 'shap_importance'] < median_negative:
                output_df.at[i, 'level_of_effect'] = 'strong'
            else:
                output_df.at[i, 'level_of_effect'] = 'weak'
    return output_df


def _grade_feature(row: pd.Series,
                   target_population: AnyStr,
                   quantiles_lower: Dict,
                   quantiles_upper: Dict,
                   col: AnyStr) -> AnyStr:
    if np.isnan(row):
        result = np.nan
    else:
        if row < quantiles_lower[col]:
            result = str(
                row) + f' (low comparing to other {target_population})'
        else:
            if row > quantiles_upper[col]:
                result = str(
                    row) + f' (high comparing to other {target_population})'
            else:
                result = str(row) + ' (around the average)'

    return result


def create_output_tables(prediction_input: pd.DataFrame,
                         prediction_output: pd.DataFrame,
                         config_name: str,
                         model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    load_config(config_name)
    pipeline = load_model(model_name)
    best_model = pipeline.named_steps.trained_model
    salesforce_df = create_results_df(prediction_output)
    result_df = pd.concat([salesforce_df[['account_id', 'rating']],
                           prediction_output['Score_1']], axis=1)
    prep_pipe = get_config('prep_pipe')
    X_pred = prep_pipe.transform(prediction_input)
    features_df = create_output_table_features(result_df=result_df,
                                               model=best_model,
                                               cols_to_drop=None,
                                               X_pred=X_pred)
    return salesforce_df, features_df
