from dataclasses import dataclass


@dataclass
class TrainTest:
    TRAIN_SIZE: float = 0.7
    DEV_TEST_RATE: float = 0.5
