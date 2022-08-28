from dataclasses import dataclass
from typing import Tuple


@dataclass
class Features:
    TARGET: str = 'Churn'
    NOT_FEATURES: Tuple = ('customerID',)
    CATEGORICAL: Tuple = (
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod',
    )
    NUMERICAL: Tuple = (
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
    )
    TO_DISCRETIZE: Tuple = ()
    DATES: Tuple = ()
