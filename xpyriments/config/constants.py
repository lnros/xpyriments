from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Constant:
    RANDOM_SEED: int = 123
    EXPERIMENTS_DIR = 'experiments'
    REPO_PATH = Path(__file__).parent.parent.resolve()
    SAVE_PATH = os.path.join(REPO_PATH, 'images')
