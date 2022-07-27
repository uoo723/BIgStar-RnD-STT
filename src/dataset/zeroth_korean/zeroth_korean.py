# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Librispeech automatic speech recognition dataset."""

from __future__ import absolute_import, division, print_function

import glob
import os

import datasets


_CITATION = """\

"""

_DESCRIPTION = """\
This is Zeroth-Korean corpus,
licensed under Attribution 4.0 International (CC BY 4.0)
The data set contains transcriebed audio data for Korean. There are 51.6 hours transcribed Korean audio for training data (22,263 utterances, 105 people, 3000 sentences) and 1.2 hours transcribed Korean audio for testing data (457 utterances, 10 people). This corpus also contains pre-trained/designed language model, lexicon and morpheme-based segmenter(morfessor).
Zeroth project introduces free Korean speech corpus and aims to make Korean speech recognition more broadly accessible to everyone.
This project was developed in collaboration between Lucas Jo(@Atlas Guide Inc.) and Wonkyum Lee(@Gridspace Inc.).

Contact: Lucas Jo(lucasjo@goodatlas.com), Wonkyum Lee(wonkyum@gridspace.com)
"""

_URL = "http://www.openslr.org/40"
_DL_URL = "https://www.openslr.org/resources/40/zeroth_korean.tar.gz"


class ZerothKoreanASRConfig(datasets.BuilderConfig):


    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(ZerothKoreanASRConfig, self).__init__(version=datasets.Version("1.0.1", ""), **kwargs)


class ZerothKoreanASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    BUILDER_CONFIGS = [
        ZerothKoreanASRConfig(name="clean", description="'Clean' speech.")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("speech", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download_and_extract(_DL_URL)
        #print(archive_path)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"archive_path": archive_path, "split_name": f"train_data_01"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"archive_path": archive_path, "split_name": f"test_data_01"}),
        ]

    def _generate_examples(self, archive_path, split_name):

        transcripts_glob = os.path.join(archive_path, split_name, "*/*/*.txt")
        for transcript_file in glob.glob(transcripts_glob):
            path = os.path.dirname(transcript_file)
            transcript_filename = os.path.basename(transcript_file)
            with open(os.path.join(path, transcript_filename)) as f:
                for line in f:
                    line = line.strip()
                    key, transcript = line.split(" ", 1)
                    audio_file = f"{key}.flac"
                    speaker_id, chapter_id = [int(el) for el in key.split("_")[:2]]
                    example = {
                        "id": key,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "file": os.path.join(path, audio_file),
                        "audio": os.path.join(path, audio_file),
                        "text": transcript,
                    }
                    yield key, example
