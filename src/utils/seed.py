import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set RNG seeds across python/numpy/torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
