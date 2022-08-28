"""
General Utils
"""

from config.constants import Constant as const
from dotenv import load_dotenv
from pathlib import Path
from pycaret.classification import load_model, load_config
from s3fs.core import S3FileSystem, S3File
from sqlalchemy.engine.url import URL
from typing import AnyStr, Optional, List, Any, Union, NoReturn

import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import sqlalchemy as sa
import time
import valohai
import warnings

# Loads config file parameters
load_dotenv()

STORAGE_OPTIONS = {
    'key': os.getenv('AWS_KEY'),
    'secret': os.getenv('AWS_SECRET'),
}
STORAGE_OPTIONS_VH = {
    'key': os.getenv('AWS_VH_KEY'),
    'secret': os.getenv('AWS_VH_SECRET'),
}

def load_data(sql_file: AnyStr,
              max_file_age_to_load: int = 7,
              load_from_query: bool = True,
              return_df: bool = True,
              is_valohai: bool = False,
              to_parquet: bool = False,
              to_csv: bool = False) -> Optional[pd.DataFrame]:
    """
    Loads data from Redshift or from file
    :param sql_file: Filename of sql query located inside sql folder
    :param max_file_age_to_load: Max number of days old a parquet file must be
                                 to use it instead of the query. The parquet
                                 file must be inside './data/'.
    :param load_from_query: Load data from the SQL query even if there is an
                            adequate file to load
    :param return_df: Whether the function returns a dataframe
    :param is_valohai: Whether the data will be loaded into Valohai
    :param to_parquet: If not Valohai, whether to save the loaded data into a
                       parquet file. If Valohai,
                       it will necessarily save a parquet file.
    :param to_csv: If not Valohai, whether to save the loaded data into a
                   csv file.
    :return: A dataframe with the output of the specified query if not Valohai
    """
    _check_type(sql_file, str)
    _check_type(max_file_age_to_load, int)
    if max_file_age_to_load < 1:
        raise ValueError('max_file_age_to_load must be positive, '
                         f'not {max_file_age_to_load}')
    _check_type(load_from_query, bool)
    _check_type(return_df, bool)
    _check_type(is_valohai, bool)
    _check_type(to_parquet, bool)
    _check_type(to_csv, bool)

    # Prepare query filepath for reading
    repo_path = os.getenv('VH_REPOSITORY_DIR') if is_valohai \
        else Path(__file__).parent.parent.resolve()
    query_full_path = os.path.join(repo_path, "sql", sql_file)
    # If no Valohai output path in the .env config file, save locally
    dir_path = os.getenv('VH_OUTPUTS_DIR', repo_path)
    dir_path = os.path.join(dir_path, 'data' if not is_valohai else '')
    parquet_file = os.path.join(dir_path,
                                sql_file.replace('sql', 'parquet'))
    csv_file = os.path.join(dir_path,
                            sql_file.replace('sql', 'csv'))

    data_file = os.path.join(
        dir_path, sql_file.replace('sql', 'parquet')) if to_parquet \
        else os.path.join(dir_path, sql_file.replace('sql', 'csv'))

    df = None
    limit_age = max_file_age_to_load * const.DAY_IN_SECONDS
    if not load_from_query:
        if os.path.isfile(data_file):
            file_age = _check_file_age_in_seconds(data_file)
            if file_age <= limit_age:
                df = pd.read_parquet(data_file) if to_parquet \
                    else pd.read_csv(data_file)
            else:
                warnings.warn(f'File is older than {max_file_age_to_load} '
                              f'days. Loading from query instead.')
        else:
            warnings.warn(f'File {data_file} does not exists. '
                          'Loading from query instead.')

    if df is None:
        url = URL.create(
            drivername='redshift+redshift_connector',
            host=os.getenv('host'),
            port=os.getenv('port'),
            database=os.getenv('dbname'),
            username=os.getenv('user'),
            password=os.getenv('password')
        )
        engine = sa.create_engine(url)

        with open(query_full_path, 'r', encoding='utf-8') as query:
            df = pd.read_sql_query(query.read(),
                                   engine).reset_index(drop=True)

    if is_valohai or to_parquet:
        df.to_parquet(parquet_file, index=False)
        filename = parquet_file.split('/')[-1]
        name = filename.split('.')[0]
        metadata_path = valohai.outputs().path(f'{filename}.metadata.json')
        tag = 'prod' if is_tag_prod() else 'dev'
        metadata = {
            'valohai.tags': [tag],
            'valohai.alias': f'{name}-{tag}',
            'key': 'value',
            'sample': 'metadata'
        }
        with open(metadata_path, 'w') as outfile:
            json.dump(metadata, outfile)
    if to_csv:
        df.to_csv(csv_file, sep=';', index=False)

    return df if return_df else None


def get_cat_features(df: pd.DataFrame) -> np.ndarray:
    """
    :param df: a dataframe
    :return: indices of categorical columns
    """
    return np.where((df.dtypes != np.float) & (df.dtypes != np.int))[0]


def get_cat_feature_names(df: pd.DataFrame) -> List[AnyStr]:
    """
    Returns the names of the categorical features in the input
    (used for Catboost)
    :param df: the input features
    :return: the names of the categorical features
    """
    return [col for col in df.columns if
            df.dtypes[col] not in [np.int, np.float]]


def save_to_valohai(obj: Any, output_name: str) -> NoReturn:
    """
    Saves any type of object with the adequate format into Valohai
    :param obj: any object
    :param output_name: name of the desired file (without format)
    """
    if isinstance(obj, pd.DataFrame):
        filename = output_name + '.parquet'
        obj.to_parquet(valohai.outputs().path(filename))
    elif isinstance(obj, (pd.Series, np.ndarray)):
        filename = output_name + '.npy'
        if isinstance(obj, pd.Series):
            obj = obj.to_numpy()
        np.save(valohai.outputs().path(filename), obj)
    else:
        filename = output_name + '.pkl'
        with open(valohai.outputs().path(filename), 'wb') as f:
            pickle.dump(obj, f)


def is_valohai_env() -> bool:
    return os.getenv('VH_REPOSITORY_DIR') == '/valohai/repository'


def _check_type(obj: Any, expected_type: Any):
    if not isinstance(obj, expected_type):
        raise TypeError(f'{obj.__name__} should be {expected_type} not '
                        f'{type(obj)}')


def _check_file_age_in_seconds(file: AnyStr) -> float:
    file_stats = os.stat(file)
    age = time.time() - file_stats.st_mtime
    return age


def unpickle(file: Union[AnyStr, S3File]) -> Any:
    if isinstance(file, S3File):
        data = pickle.loads(file.read())
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def valohai_plot_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', help='chooses which site to plot')
    parser.add_argument('--plot_past_n_days', default=60,
                        help='chooses the time window to plot')

    return parser.parse_args()


def valohai_slack_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staging',
                        help='flags if the message should be send to '
                             'the staging channel',
                        action='store_true')

    return parser.parse_args()


def read_from_s3(filename: str, valohai=False):
    funcs = {
        'csv': pd.read_csv,
        'parquet': pd.read_parquet,
        'npy': np.load,
        'pkl': unpickle,
    }
    file_type = filename.split('.')[-1]
    if not valohai:
        full_path = os.path.join('s3://',
                                 os.getenv('S3_BUCKET'),
                                 os.getenv('S3_PATH'),
                                 filename)
        storage_options = STORAGE_OPTIONS
    else:
        full_path = os.path.join('s3://',
                                 os.getenv('S3_VH_BUCKET'),
                                 filename)
        storage_options = STORAGE_OPTIONS_VH

    try:
        obj_ = funcs[file_type](full_path, storage_options=storage_options)
    except TypeError:
        s3 = S3FileSystem(**storage_options)
        obj_ = funcs[file_type](s3.open(full_path))

    return obj_


def is_tag_prod():
    try:
        with open('/valohai/config/execution.json') as json_file:
            data = json.load(json_file)
            if data.get('valohai.creator-name') == const.CREATOR_NAME_WHEN_TRIGGER:
                return True
            all_tags = [data.get('valohai.pipeline-tags', []),
                        data.get('valohai.execution-tags', [])]
            all_tags_flatten = [tag for tags_list in all_tags for tag in tags_list]
            if 'prod' in all_tags_flatten:
                return True
            return False
    except FileNotFoundError:
        return False
