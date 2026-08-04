# common/utils/seed.py
import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """再現性確保のため、各種乱数シードを一括固定する"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorchを使用している場合の処理（入っていなくてもエラーにならないようtry-except）
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass