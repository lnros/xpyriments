from kmodes.kmodes import KModes
from shap import TreeExplainer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cityblock, correlation, cosine, euclidean
import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import warnings

from utils.general_utils import get_cat_features_names


# --- check how similar the predictions to the actual class in the training
# --- check maybe better using class 1 instead of predicted class 1

def what_if_analysis_one_record(account_c, X_test, X_train_what_if_numeric, X_test_what_if_numeric,
                                train_cluster, test_cluster, shap_df_train, shap_df_test, shap_threshold=0.05,
                                n_features_to_analyze=2, distance_function='manhattan', beta_feature_diff=0):
    """
    This function produce a what-if analysis for a given id.
    The test sample is compared with the train data from the same cluster:
    1. We find the closet sample.
    2. We compute both the SHAP difference & feature's values difference between the test sample to the closet sample.
    3. We aggregate both differences (for each feature).
    4. We choose the feature to analyze by the aggregated maximum difference (if it SHAP difference is bigger the the shap_threshold)
    5. If the number of features that analyzed is smaller than 'n_features_to_analyze' repeat step 4.

    :param account_c: The account to analyze
    :param X_test: the test data
    :param X_train_what_if_numeric: train data with the numeric features that included in the what-if analysis
    :param X_test_what_if_numeric: test data with the numeric features that included in the what-if analysis
    :param train_cluster: a numeric vector with the cluster label for the train data
    :param test_cluster: a numeric vector with the cluster label for the test data
    :param shap_df_train: SHAP values for the training data
    :param shap_df_test: SHAP values for the test data
    :param shap_threshold: min SHAP difference that included in the analysis
    :param n_features_to_analyze: max number of features to analyze
    :param distance_function: the distance function
    :param feature_diff_weight: the weight for the difference between the feature value
    :return:
    """
    shap_test_sample = np.array(shap_df_test.loc[account_c])

    X_train_curr = X_train_what_if_numeric[train_cluster == test_cluster.loc[account_c]]
    data_for_scaling = pd.concat([X_train_curr, pd.DataFrame(X_test_what_if_numeric.loc[account_c]).transpose()],
                                 axis=0)
    data_for_scaling = pd.DataFrame(StandardScaler().fit_transform(data_for_scaling),
                                    columns=data_for_scaling.columns, index=data_for_scaling.index)
    scaled_train, scaled_sample = data_for_scaling.drop(account_c, axis=0), data_for_scaling.loc[account_c]
    # - consider change to sickit-learn
    if distance_function.lower() == 'manhattan':
        dists = [cityblock(scaled_sample, scaled_train.iloc[i]) for i in
                 (range(scaled_train.shape[0]))]
    elif distance_function.lower() == 'correlation':
        dists = [correlation(scaled_sample, scaled_train.iloc[i]) for i in
                 (range(scaled_train.shape[0]))]
    elif distance_function.lower() == 'cosine':
        dists = [cosine(scaled_sample, scaled_train.iloc[i]) for i in
                 (range(scaled_train.shape[0]))]
    elif distance_function.lower() == 'euclidean':
        dists = [euclidean(scaled_sample, scaled_train.iloc[i]) for i in
                 (range(scaled_train.shape[0]))]
    else:
        raise ValueError(distance_function + ' is not a valid distance function')
    closest_obs = X_train_curr.iloc[[np.argmin(dists)]]
    closest_obs_scaled = scaled_train.iloc[[np.argmin(dists)]].values[0]
    ##features_diff = [euclidean(closest_obs_scaled[i], scaled_sample.values[i]) for i in range(closest_obs_scaled.shape[0])]
    ##features_diff = np.array([x if x != 0 else np.nan for x in features_diff])
    shap_closet_obs = np.array(shap_df_train.loc[closest_obs.index])
    shap_diff = np.subtract(shap_closet_obs, shap_test_sample).reshape(-1)
    features_diff = np.subtract(closest_obs_scaled, scaled_sample)
    features_diff = np.array([x if x > 0 else np.nan for x in features_diff])
    if not np.all(np.isnan(features_diff)):
        scaler = MinMaxScaler(feature_range=(np.finfo(float).eps, 1))
        dists_df = pd.DataFrame()
        dists_df['shap_dist'] = shap_diff
        dists_df['feature_dist'] = 1 - features_diff
        dists_df = pd.DataFrame(scaler.fit_transform(dists_df), columns=dists_df.columns)
        dists_df.loc[list(np.where(np.array(features_diff) == 0)[0]), 'feature_dist'] = np.nan

        dists_df.loc[dists_df.feature_dist == 0, 'feature_dist'] = np.finfo(float).eps
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in multiply')
            combined_diff = ((1 + beta_feature_diff ** 2) * dists_df['shap_dist'] * dists_df['feature_dist']) / \
                            (beta_feature_diff ** 2 * dists_df['shap_dist'] + dists_df['feature_dist'])
        max_diff_loc = np.nanargmax(combined_diff)

        count_found_features = 0
        what_if_df = pd.DataFrame(
            columns=['account_id', 'feature_name', 'feature_value', 'neighbor_feature_value', 'neighbor_id'])
        # consider sort the shap diff / combined_diff
        while (count_found_features < n_features_to_analyze) & (shap_diff[max_diff_loc] > shap_threshold):
            feature_name = list(scaled_train.columns)[max_diff_loc]
            curr_val = X_test.loc[account_c, feature_name]
            nb_val = closest_obs[feature_name].values[0]
            if nb_val > curr_val:
                row_vals = [account_c, feature_name, curr_val, nb_val, closest_obs.index[0]]
                what_if_df.loc[count_found_features] = row_vals
                count_found_features += 1
            combined_diff[np.nanargmax(combined_diff)] = np.nan
            if not np.all(np.isnan(combined_diff)):
                max_diff_loc = np.nanargmax(combined_diff)
            else:
                break
            return what_if_df


def what_if_analysis(model, X_train, X_test, what_if_features, n_clusters=5, prob_threshold=0.5, shap_threshold=0.05,
                     n_features_to_analyze=2, distance_function='manhattan', beta_feature_diff=0):
    # like what if features (numeric) add cat features for kmodes
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    X_train_what_if = X_train[train_scores > prob_threshold]
    X_test_what_if = X_test[test_scores < prob_threshold]

    cat_cols = get_cat_features_names(X_train)
    cat_cols = [col for col in cat_cols if col != 'leading_tech']
    df_kmodes = pd.concat([X_train_what_if[cat_cols], X_test_what_if[cat_cols]], axis=0)
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0, n_jobs=-1, max_iter=5000, random_state=2)
    km.fit(df_kmodes)

    # add function that predicts and create the series
    train_cluster = km.predict(X_train_what_if[cat_cols])
    train_cluster = pd.Series(train_cluster, index=X_train_what_if.index)
    test_cluster = km.predict(X_test_what_if[cat_cols])
    test_cluster = pd.Series(test_cluster, index=X_test_what_if.index)

    X_train_what_if_numeric = X_train_what_if.select_dtypes(include=np.number)[what_if_features]
    X_test_what_if_numeric = X_test_what_if.select_dtypes(include=np.number)[what_if_features]

    # split to functions (e.g., shap function. maybe class)
    explainer = TreeExplainer(model)
    shap_df_train = pd.DataFrame(explainer.shap_values(X_train), columns=X_train.columns, index=X_train.index)
    shap_df_test = pd.DataFrame(explainer.shap_values(X_test), columns=X_test.columns, index=X_test.index)

    shap_df_train = shap_df_train.loc[X_train_what_if.index, what_if_features]
    shap_df_test = shap_df_test.loc[X_test_what_if.index, what_if_features]

    what_if_df = Parallel(n_jobs=os.cpu_count())(delayed(what_if_analysis_one_record)(account_c, X_test,
                                                                                      X_train_what_if_numeric,
                                                                                      X_test_what_if_numeric,
                                                                                      train_cluster, test_cluster,
                                                                                      shap_df_train, shap_df_test,
                                                                                      shap_threshold,
                                                                                      n_features_to_analyze,
                                                                                      distance_function,
                                                                                      beta_feature_diff)
                                                 for account_c in X_test_what_if.index)
    what_if_df = pd.concat(what_if_df).reset_index(drop=True)
    return what_if_df


def evaluate_what_if(X_test, df_what_if, model, neighbor_value=False, round_up=0.2):
    X_test_modified = X_test.copy()
    for i in df_what_if.index:
        curr_row = df_what_if.loc[i]
        if neighbor_value:
            X_test_modified.loc[curr_row.account_id, curr_row.feature_name] = curr_row.neighbor_feature_value
        else:
            X_test_modified.loc[curr_row.account_id, curr_row.feature_name] = \
                np.round(X_test_modified.loc[curr_row.account_id, curr_row.feature_name] * (1 + round_up))

    old_scores = model.predict_proba(X_test.loc[df_what_if.account_id.unique()])[:, 1]
    new_scores = model.predict_proba(X_test_modified.loc[df_what_if.account_id.unique()])[:, 1]
    p_changes_vector = (new_scores - old_scores) / old_scores
    return np.mean(p_changes_vector), np.mean(p_changes_vector > 0)
