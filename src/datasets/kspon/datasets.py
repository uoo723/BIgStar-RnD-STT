"""
Created on 2022/06/07
@author Sangwoo Han
"""
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor


class KSponSpeechDataset(Dataset):
    """Dataset for KSponSpeech from AI Hub

    https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123

    Args:
        filepath (str): Preprocessed data path from `python main.py preprocess-kspon`
        original_path (str, optional): Original data path residing audio files.
    """

    def __init__(self, filepath: str, original_path: Optional[str] = None) -> None:
        self.df = pd.read_csv(filepath)
        self.original_path = original_path or os.path.join(
            os.path.dirname(os.path.dirname(filepath)), "original"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        audio_path = os.path.join(self.original_path, self.df.iloc[idx]["audio_path"])
        transcript = self.df.iloc[idx]["transcript"]

        signal = np.memmap(audio_path, dtype="h", mode="r").astype("float32")
        signal = np.array(signal)

        return signal, len(signal), transcript


def dataloader_collate_fn(
    batch: Iterable[Tuple[np.ndarray, int, str]],
    processor: Wav2Vec2Processor,
    sampling_rate: int = 16_000,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    signals = [b[0] for b in batch]
    transcripts = [b[2] for b in batch]

    signal = processor(
        signals, sampling_rate=sampling_rate, return_tensors="pt", padding="longest"
    )

    with processor.as_target_processor():
        transcript_inputs: Dict[str, torch.Tensor] = processor(
            transcripts, return_tensors="pt", padding="longest"
        )
        transcript_inputs["lengths"] = transcript_inputs["attention_mask"].sum(axis=-1)

    return signal, transcript_inputs
