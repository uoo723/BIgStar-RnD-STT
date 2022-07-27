"""
Created on 2022/06/07
@author Sangwoo Han
"""
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jiwer import cer as compute_cer
from jiwer import wer as compute_wer
from logzero import logger
from optuna import Trial
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoProcessor, Wav2Vec2Config, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput

from .. import base_trainer
from ..base_trainer import BaseTrainerModel, get_ckpt_path, load_model_hparams
from ..dataset.kspon import KSponSpeechDataset, dataloader_collate_fn
from ..utils import AttrDict, copy_file, filter_arguments, get_num_batches

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class Wav2VecTrainerModel(BaseTrainerModel):
    MODEL_HPARAMS: Iterable[str] = [
        "pretrained_model_name",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "hidden_size",
        "ctc_loss_reduction",
        "ctc_zero_infinity",
    ]

    def __init__(
        self,
        pretrained_model_name: str = "kresnik/wav2vec2-large-xlsr-korean",
        dataset_filepath: str = "./data/kspon_speech/preprocessed/transcripts.csv",
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
        self.dataset_filepath = dataset_filepath
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    @property
    def model_hparams(self) -> Iterable[str]:
        return Wav2VecTrainerModel.MODEL_HPARAMS

    def prepare_data(self) -> None:
        pass

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        if stage == "predict":
            raise ValueError(f"{stage} stage is not supported")

        if self.train_dataset is None:
            dataset = KSponSpeechDataset(self.dataset_filepath)

            self.train_ids, self.valid_ids = train_test_split(
                np.arange(len(dataset)),
                test_size=self.valid_size,
                random_state=self.seed,
            )

            self.train_dataset = Subset(dataset, self.train_ids)
            self.val_dataset = Subset(dataset, self.valid_ids)

        if self.test_dataset is None:
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(dataloader_collate_fn, processor=self.processor),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(dataloader_collate_fn, processor=self.processor),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(dataloader_collate_fn, processor=self.processor),
        )

    def setup_model(self, stage: Optional[str] = None) -> None:
        if self.model is not None:
            return

        if self.run_id is not None:
            hparams = load_model_hparams(self.log_dir, self.run_id, self.model_hparams)
        else:
            hparams = {param: getattr(self, param) for param in self.model_hparams}

        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name)
        self.model = Wav2Vec2ForCTC(
            Wav2Vec2Config(
                vocab_size=self.processor.tokenizer.vocab_size,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                **filter_arguments(hparams, Wav2Vec2Config),
            )
        )
        # self.loss_fn = nn.CTCLoss(blank=self.processor.tokenizer.pad_token_id)

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        batch_signal, batch_transcript = batch
        batch_size = len(batch_signal["input_values"])

        inputs = filter_arguments(batch_signal, self.model.forward)
        outputs: CausalLMOutput = self.model(
            **inputs, labels=batch_transcript["input_ids"]
        )

        # signal_lengths = torch.full(
        #     size=(batch_size,), fill_value=logits.shape[1], dtype=torch.long
        # ).to(self.device)

        # loss = self.loss_fn(
        #     logits.log_softmax(dim=-1).transpose(0, 1),
        #     batch_transcript["input_ids"],
        #     signal_lengths,
        #     # batch_transcript["lengths"],
        # )

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

        # logger.debug("pred_ids:")
        # print(pred_ids)

        if is_val:
            # batch_size = len(batch_signal["input_values"])
            # signal_lengths = torch.full(
            #     size=(batch_size,), fill_value=logits.shape[1], dtype=torch.long
            # ).to(self.device)
            # loss = self.loss_fn(
            #     logits.log_softmax(dim=-1).transpose(0, 1),
            #     batch_transcript["input_ids"],
            #     signal_lengths,
            #     # batch_transcript["lengths"],
            # )
            self.log("loss/val", outputs.loss)

        return pred

    def _validation_and_test_epoch_end(
        self, outputs: EPOCH_OUTPUT, is_val: bool = True
    ) -> None:
        predictions = [p2 for p1 in outputs for p2 in p1]
        gt = self.val_dataset.dataset.df.iloc[self.valid_ids][: len(predictions)][
            "transcript"
        ].tolist()

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
    valid_dataset_name = ["dataset"]
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
    assert args.mode == "predict", "mode must be predict"
    assert args.run_id is not None, "run_id must be specified"
    assert args.submission_output is not None, "submission output must be specified"

    logger.info(f"run_id: {args.run_id}")
    logger.info(f"submission_output: {args.submission_output}")

    if args.topk_filepath:
        logger.info(f"topk_filepath: {args.topk_filepath}")

    os.makedirs(os.path.dirname(args.submission_output), exist_ok=True)

    if os.path.exists(args.submission_output):
        if args.silent:
            return

        if not args.overwrite:
            click.confirm(
                f"{os.path.basename(args.submission_output)} is already existed."
                " Overwrite it?",
                abort=True,
            )

    Path(args.submission_output).touch()

    ############################# Save runscript #######################################
    os.makedirs(os.path.dirname(args.submission_output), exist_ok=True)
    if args.run_script:
        dirname = os.path.dirname(args.submission_output)
        basename, ext = os.path.splitext(os.path.basename(args.run_script))
        basename += "_" + os.path.splitext(os.path.basename(args.submission_output))[0]
        copy_file(args.run_script, os.path.join(dirname, basename + ext))
    ####################################################################################

    ################################# Load Data ########################################
    logger.info("Load Data...")
    _, test_data, test_question = load_data(args.data_dir)
    _, test_docs = preprocess_data(test_data, return_query_to_docs=False)
    test_queries = dict(
        zip(test_question["question_id"], test_question["question_text"])
    )

    if args.topk_filepath:
        df = pd.read_csv(args.topk_filepath)
        test_candidates = dict(
            zip(df["question_id"], [d.split(",") for d in df["paragraph_id"]])
        )
    else:
        test_query_id_i2s = dict(zip(range(len(test_queries)), test_queries.keys()))
        tsv_path = os.path.join(args.data_dir, "top1000", "test_top1000_00.txt")
        df = pd.read_csv(tsv_path, sep=" ", header=None)
        df[0] = df[0].map(lambda x: test_query_id_i2s[x])
        test_candidates: Dict[str, List[str]] = df.groupby(0)[2].apply(list).to_dict()
    ####################################################################################

    ################################## Load Model ######################################
    logger.info("Load Model...")
    hparams = load_model_hparams(
        args.log_dir, args.run_id, Wav2VecTrainerModel.MODEL_HPARAMS
    )

    model = MonoBERT(**filter_arguments(hparams, MonoBERT))

    ckpt_path = get_ckpt_path(args.log_dir, args.run_id, load_best=not args.load_last)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    swa_callback_key = None
    callbacks: Dict[str, Any] = ckpt["callbacks"]
    for key in callbacks.keys():
        if "StochasticWeightAveraging" in key:
            swa_callback_key = key
            break

    state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]

    if swa_callback_key is not None and "average_model" in callbacks[swa_callback_key]:
        logger.info("Use averaged weights")
        avg_state_dict: Dict[str, torch.Tensor] = callbacks[swa_callback_key][
            "average_model"
        ]
        avg_state_dict.pop("models_num")
        state_dict.update(avg_state_dict)

    state_dict = OrderedDict(
        zip(
            [key.replace("model.", "") for key in state_dict.keys()],
            state_dict.values(),
        )
    )

    model.load_state_dict(state_dict)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(hparams["pretrained_model_name"])
    ####################################################################################

    ################################## Inference #######################################
    logger.info("Start Inference")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    batch_size = args.test_batch_size
    max_length = args.max_length
    topk = args.topk_candidates
    answers = []

    model.eval()
    for q_id, doc_ids in tqdm(test_candidates.items(), desc="inference..."):
        query_str = test_queries[q_id]
        doc_ids = np.array(doc_ids)
        num_batches = get_num_batches(batch_size, topk)
        predictions = []
        for b in range(num_batches):
            doc_str = [
                test_docs[d_id]
                for d_id in doc_ids[:topk][b * batch_size : (b + 1) * batch_size]
            ]
            inputs: Dict[str, torch.Tensor] = tokenizer(
                [query_str] * len(doc_str),
                doc_str,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation="longest_first",
            )
            with torch.no_grad():
                outputs: torch.Tensor = model(
                    {k: v.to(args.device) for k, v in inputs.items()}
                )
            predictions.append(outputs.cpu())
        rank = np.concatenate(predictions).argsort()[::-1]
        answers.append(",".join(doc_ids[:topk][rank][: args.final_topk]))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ####################################################################################

    ############################### Make Submission ####################################
    logger.info("Make submission")
    submission = pd.DataFrame(
        data={"question_id": test_question["question_id"], "paragraph_id": answers}
    )
    submission.to_csv(args.submission_output, index=False)
    logger.info(f"Saved into {args.submission_output}")
    ####################################################################################

    return submission
