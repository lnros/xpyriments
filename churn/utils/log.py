"""
Logging class using YAML config file
"""
from churn.config.constants import Constant as const

import logging
import logging.config
import yaml


class Log:
    """
    Logging class using YAML config file
    """

    def __init__(self, log_cfg_file=f'{const.REPO_PATH}/'
                                    'churn/config/logger.yaml'):
        self.logger = None
        self.log_cfg_file = log_cfg_file

    def start_logging(self):
        with open(self.log_cfg_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logger = logging.getLogger(__name__)
        self.logger = logger
