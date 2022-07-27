"""
Created on 2022/06/09
@author Sangwoo Han
@ref https://github.com/sooftware/kospeech/blob/latest/dataset/kspon/preprocess/character.py
"""
import os

import pandas as pd
from logzero import logger
from tqdm.auto import tqdm
from collections import Counter


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    # freq_list = ch_labels["freq"]

    for id_, char in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue

    return target[:-1]


def generate_character_labels(transcripts, labels_dest):
    logger.info("create_char_labels started..")

    label_freq = Counter()

    for transcript in transcripts:
        for ch in transcript:
            label_freq[ch] += 1

    # sort together Using zip
    label_freq, label_list = zip(
        *sorted(zip(label_freq.values(), label_freq.keys()), reverse=True)
    )
    label = {"id": [0, 1, 2], "char": ["<pad>", "<sos>", "<eos>"], "freq": [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label["id"].append(idx + 3)
        label["char"].append(ch)
        label["freq"].append(freq)

    label["id"] = label["id"][:2000]
    label["char"] = label["char"][:2000]
    label["freq"] = label["freq"][:2000]

    label_df = pd.DataFrame(label)
    label_df.to_csv(
        os.path.join(labels_dest, "aihub_labels.csv"), encoding="utf-8", index=False
    )


def generate_character_script(audio_paths, transcripts, save_path):
    logger.info("create_script started...")

    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame(data={"audio_path": audio_paths, "transcript": transcripts})
    df.to_csv(os.path.join(save_path, "transcripts.csv"), index=False)
