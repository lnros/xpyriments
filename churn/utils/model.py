from pycaret.classification import *
from sklearn.metrics import accuracy_score, average_precision_score, \
    confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from config.constants import Constant as const
from config.features import Features
from config.model_selection import TrainTest
from utils.general_utils import save_to_valohai, is_tag_prod

import json
import logging
import os
import valohai


def initialize_train_setup(data: pd.DataFrame):
    df_train, df_test = train_test_split(data,
                                         train_size=TrainTest.TRAIN_SIZE,
                                         random_state=const.RANDOM_SEED,
                                         stratify=data['class'],
                                         )
    churn_setup = setup(
        experiment_name='enterprise_churn',
        data=df_train,
        test_data=df_test,
        target=Features.TARGET,
        categorical_features=list(Features.CATEGORICAL),
        numeric_features=list(Features.NUMERICAL),
        date_features=list(Features.DATES),
        ignore_features=list(Features.NOT_FEATURES),
        session_id=const.RANDOM_SEED,
        normalize=False,
        fix_imbalance=False,
        feature_selection=True,
        feature_selection_method='boruta',
        html=False,
        silent=True,
    )
    filename = 'enterprise_churn_cfg.pkl'
    metadata_path = valohai.outputs().path(f'{filename}.metadata.json')
    tag = 'prod' if is_tag_prod() else 'dev'
    metadata = {
        'valohai.tags': [tag],
        'valohai.alias': f'config-{tag}',
        'key': 'value',
        'sample': 'metadata'
    }
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)
    save_config(f"{os.getenv('VH_OUTPUTS_DIR')}/{filename}")


def train_model(config_name):
    load_config(config_name)
    best_model = compare_models(
        include=['rf', 'lightgbm', 'xgboost', 'catboost', 'ada'],
        sort='F1'
    )
    filename = 'best_model.pkl'
    metadata_path = valohai.outputs().path(f'{filename}.metadata.json')
    tag = 'prod' if is_tag_prod() else 'dev'
    metadata = {
        'valohai.tags': [tag],
        'valohai.alias': f'model-{tag}',
        'key': 'value',
        'sample': 'metadata'
    }
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)
    save_model(best_model, f"{os.getenv('VH_OUTPUTS_DIR')}/best_model")
    y_test = get_config('y_test')
    X_test = get_config('X_test')
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)[:, 1]
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
            'model_name': [type(best_model).__name__],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1],
            'AP': [ap],
        }
    )
    # logging into valohai
    with valohai.logger() as vh_log:
        for col in generalization_results.columns:
            try:
                metric = generalization_results[col].values[0]
            except Exception as e:
                logging.exception(e)
            vh_log.log(col, metric)


def predict(config_name, model_name, data):
    load_config(config_name)
    pipeline = load_model(model_name)
    model = pipeline.named_steps.trained_model
    try:
        predict_df = predict_model(model, data=data, raw_score=True)
    except Exception as e:
        logging.exception(e)
    else:
        with valohai.logger() as vh_log:
            vh_log.log('prediction_shape', predict_df.shape)
            vh_log.log(
                'prediction_distribution',
                predict_df['Label'].value_counts(normalize=True).to_json()
            )
        filename = 'predict_output.parquet'
        name = filename.split('.')[0]
        metadata_path = valohai.outputs().path(f'{filename}.metadata.json')
        tag = 'prod' if is_tag_prod() else 'dev'
        metadata = {
            'valohai.tags': [tag],
            'valohai.alias': f'prediction-{tag}',
            'key': 'value',
            'sample': 'metadata'
        }
        with open(metadata_path, 'w') as outfile:
            json.dump(metadata, outfile)
        save_to_valohai(predict_df, name)
