"""Reproducibility - unified seeding."""
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Fix the seed for python / numpy / (optionally) torch at once.

    torch is imported lazily so this function can be imported even in
    environments without torch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not available - ignore.
        pass
