from dataclasses import dataclass


@dataclass
class Constant:
    RANDOM_SEED: int = 123
    LOW_THRESHOLD: float = 0.1
    MEDIUM_THRESHOLD: float = 0.3
    HIGH_THRESHOLD: float = 0.5
    CREATOR_NAME_WHEN_TRIGGER: str = 'JFrog'
    DAY_IN_SECONDS: int = 24 * 60 * 60
