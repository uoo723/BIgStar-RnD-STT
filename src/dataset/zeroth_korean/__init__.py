"""
Created on 2022/06/07
@author Sangwoo Han
"""
from typing import Tuple
import os
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from . import zeroth_korean

LOADER_PATH = os.path.abspath(zeroth_korean.__file__)


class ZerothKoreanDataset(Dataset):
    """Dataset for Zeroth Korean

    https://www.openslr.org/40/

    Args:
        cache_dir (str): Cache directory where dataset is downloaded and preprocessed.
    """

    def __init__(self, cache_dir: str) -> None:
        self.dataset = load_dataset(LOADER_PATH, cache_dir=cache_dir, split="test")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        signal = self.dataset[idx]["audio"]["array"]
        transcript = self.dataset[idx]["text"]
        return signal, len(signal), transcript
