"""
Created on 2022/06/07
@author Sangwoo Han
"""
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from jiwer import cer as compute_cer
from jiwer import wer as compute_wer
from optuna import Trial
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, Wav2Vec2Config, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..dataset.kspon import KSponSpeechDataset, dataloader_collate_fn
from ..dataset.zeroth_korean import ZerothKoreanDataset
from ..utils import AttrDict, filter_arguments

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class Wav2VecTrainerModel(BaseTrainerModel):
    MODEL_HPARAMS: Iterable[str] = BaseTrainerModel.MODEL_HPARAMS + [
        "pretrained_model_name",
        "use_pretrained_model",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "hidden_size",
    ]

    def __init__(
        self,
        pretrained_model_name: str = "kresnik/wav2vec2-large-xlsr-korean",
        use_pretrained_model: bool = False,
        dataset_filepath: str = "./data/kspon_speech/preprocessed/transcripts.csv",
        cache_dir: str = "./data/huggingface/datasets",
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_size: int = 768,
        ctc_loss_reduction: str = "sum",
        ctc_zero_infinity: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pretrained_model_name = pretrained_model_name
        self.use_pretrained_model = use_pretrained_model
        self.dataset_filepath = dataset_filepath
        self.cache_dir = cache_dir
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    def prepare_data(self) -> None:
        ZerothKoreanDataset(cache_dir=self.cache_dir)

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        if stage == "predict":
            raise ValueError(f"{stage} stage is not supported")

        if stage == "fit" and self.train_dataset is None:
            if self.dataset_name == "kspon":
                dataset = KSponSpeechDataset(self.dataset_filepath)
            else:
                dataset = ZerothKoreanDataset(cache_dir=self.cache_dir, split="train")

            self.train_ids, self.valid_ids = train_test_split(
                np.arange(len(dataset)),
                test_size=self.valid_size,
                random_state=self.seed,
            )

            self.train_dataset = Subset(dataset, self.train_ids)
            self.val_dataset = Subset(dataset, self.valid_ids)

        if stage == "test" and self.test_dataset is None:
            self.test_dataset = ZerothKoreanDataset(
                cache_dir=self.cache_dir, split="test"
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
            collate_fn=partial(dataloader_collate_fn, processor=self.processor),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
            collate_fn=partial(dataloader_collate_fn, processor=self.processor),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
            collate_fn=partial(dataloader_collate_fn, processor=self.processor),
        )

    def setup_model(self, stage: Optional[str] = None) -> None:
        if self.model is not None:
            return

        hparams = {param: getattr(self, param) for param in self.MODEL_HPARAMS}

        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name)

        if self.use_pretrained_model:
            self.model = Wav2Vec2ForCTC.from_pretrained(self.pretrained_model_name)
        else:
            self.model = Wav2Vec2ForCTC(
                Wav2Vec2Config(
                    vocab_size=self.processor.tokenizer.vocab_size,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    **filter_arguments(hparams, Wav2Vec2Config),
                )
            )

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        batch_signal, batch_transcript = batch

        inputs = filter_arguments(batch_signal, self.model.forward)
        outputs: CausalLMOutput = self.model(
            **inputs, labels=batch_transcript["input_ids"]
        )

        self.log("loss/train", outputs.loss)
        return outputs.loss

    def _validation_and_test_step(
        self, batch: BATCH, is_val: bool = True
    ) -> Optional[STEP_OUTPUT]:
        batch_signal, batch_transcript = batch

        inputs = filter_arguments(batch_signal, self.model.forward)
        outputs: CausalLMOutput = self.model(
            **inputs, labels=batch_transcript["input_ids"]
        )
        pred_ids = outputs.logits.argmax(dim=-1)
        pred = self.processor.batch_decode(pred_ids)

        if is_val:
            self.log("loss/val", outputs.loss)

        return pred

    def _validation_and_test_epoch_end(
        self, outputs: EPOCH_OUTPUT, is_val: bool = True
    ) -> None:
        predictions = [p2 for p1 in outputs for p2 in p1]

        if is_val:
            df: pd.DataFrame = self.val_dataset.dataset.df
            gt = df["transcript"][self.valid_ids][: len(predictions)].tolist()
        else:
            df: pd.DataFrame = self.test_dataset.df
            gt = df["transcript"][: len(predictions)].tolist()

        cer = compute_cer(gt, predictions)
        wer = compute_wer(gt, predictions)

        if is_val:
            self.log_dict({"val/cer": cer, "val/wer": wer}, prog_bar=True)
        else:
            self.log_dict({"test/cer": cer, "test/wer": wer})

    def validation_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=True)

    def test_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=False)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=True)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=False)


def check_args(args: AttrDict) -> None:
    valid_early_criterion = ["cer", "wer", "loss"]
    valid_model_name = ["Wav2Vec"]
    valid_dataset_name = ["kspon", "zeroth_korean"]
    base_trainer.check_args(
        args, valid_early_criterion, valid_model_name, valid_dataset_name
    )


def init_run(args: AttrDict) -> None:
    base_trainer.init_run(args)


def train(
    args: AttrDict,
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
) -> Tuple[float, pl.Trainer]:
    return base_trainer.train(
        args,
        Wav2VecTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        Wav2VecTrainerModel,
        metrics=["cer", "wer"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def predict(args: AttrDict) -> Any:
    pass
