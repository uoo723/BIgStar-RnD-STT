"""
Created on 2022/06/07
@author Sangwoo Han
"""
import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from transformers import Wav2Vec2Processor

from . import zeroth_korean

LOADER_PATH = os.path.abspath(zeroth_korean.__file__)


def load_zeroth_korean_dataset(
    cache_dir: str,
    split: Optional[str] = None,
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    return load_dataset(LOADER_PATH, cache_dir=cache_dir, split=split)


def dataloader_collate_fn(
    batch: Iterable[Dict[str, Any]],
    processor: Wav2Vec2Processor,
    sampling_rate: int = 16_000,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    signals = [b["audio"]["array"] for b in batch]
    transcripts = [b["text"] for b in batch]

    signal = processor(
        signals, sampling_rate=sampling_rate, return_tensors="pt", padding="longest"
    )

    with processor.as_target_processor():
        transcript_inputs: Dict[str, torch.Tensor] = processor(
            transcripts, return_tensors="pt", padding="longest"
        )
        transcript_inputs["lengths"] = transcript_inputs["attention_mask"].sum(axis=-1)

    return signal, transcript_inputs
