"""
Created on 2022/06/07
@author Sangwoo Han
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset

from . import zeroth_korean

LOADER_PATH = os.path.abspath(zeroth_korean.__file__)


class ZerothKoreanDataset(torch.utils.data.Dataset):
    """Dataset for Zeroth Korean

    https://www.openslr.org/40/

    Args:
        cache_dir (str): Cache directory where dataset is downloaded and preprocessed.
    """

    def __init__(self, cache_dir: str, split: str = "train") -> None:
        self.df: pd.DataFrame = (
            load_dataset(LOADER_PATH, cache_dir=cache_dir, split=split)
            .rename_column("text", "transcript")
            .to_pandas()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        signal = sf.read(self.df.iloc[idx]["audio"]["path"])[0]
        transcript = self.df.iloc[idx]["transcript"]
        return signal, len(signal), transcript
