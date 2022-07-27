"""
Created on 2022/06/13
@author Sangwoo Han
"""
from typing import Any

import click

from main import cli
from src.dataset.kspon.preprocess import preprocess_data
from src.dataset.kspon.preprocess.character import (
    generate_character_script,
)
from src.utils import AttrDict, log_elapsed_time


@cli.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default="./data/kspon_speech/original",
    help="Data path of original KSponSpeech dataset",
)
@click.option(
    "--save-path",
    type=click.Path(),
    default="./data/kspon_speech/preprocessed",
    help="Path to be saved preprocseed data",
)
@log_elapsed_time
def preprocess_kspon(**args: Any):
    """Preprocess KsponSpeech"""
    args = AttrDict(args)

    audio_paths, transcripts = preprocess_data(args.data_dir)
    generate_character_script(audio_paths, transcripts, args.save_path)
