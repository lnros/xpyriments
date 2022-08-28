from dataclasses import dataclass
from typing import Tuple


@dataclass
class Features:
    TARGET: str = 'class'
    NOT_FEATURES: Tuple = (
        'account_id',
        'account_name',
        'opportunity_name',
        'opportunity_created_date',
        'expiry_date',
        'new_n_servers',
        'new_product',
        'account_type',
        'top_subscription',
        'is_downsell_servers',
        'is_downsell_pro',
        'is_lost',
        'relevant_date',
        # 'n_ha_servers',
        # 'n_trials',
        # 'n_ent_trials',
        # 'n_sessions_past_year',
        # 'n_xray_sessions_past_year',
        # 'n_performance_mentioned_cases',
        # 'n_bal_mentioned_cases',
        # 'n_ent_mentioned_cases',
        # 'n_ha_mentioned_cases',
        # 'n_mul_mentioned_cases',
        # 'n_rep_mentioned_cases',
    )
    CATEGORICAL: Tuple = (
        'territory',
        'is_artifactory_7_0',
        'original_product',
    )
    NUMERICAL: Tuple = (
        'pull_replications',
        'push_replications',
        'event_replications',
        'cases_within_3_last_months',
        'cases_within_last_year',
        'n_security_subject',
        'n_ha_subject',
        'n_docker_subject',
        'using_ha',
        # 'n_ha_servers',
        'n_sessions_past_year',
        'n_xray_sessions_past_year',
        # 'n_trials',
        # 'n_ent_trials',
        'n_performance_mentioned_cases',
        'n_bal_mentioned_cases',
        'n_ent_mentioned_cases',
        'n_ha_mentioned_cases',
        'n_mul_mentioned_cases',
        'n_rep_mentioned_cases',
    )
    TO_DISCRETIZE: Tuple = ()
    DATES: Tuple = ()
