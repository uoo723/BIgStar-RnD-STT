# 대스타 R&D

```bash
├── main.py
├── preprocess.py
├── train.py
├── scripts
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   ├── run_wav2vec_kspon.sh
│   ├── run_wav2vec_zeroth.sh
│   ├── run_test.sh
├── src
│   ├── base_trainer.py
│   ├── callbacks.py
│   ├── optimizers.py
│   ├── utils.py
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── kspon
│   │   │   ├── __init__.py
│   │   │   └── preprocess
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       └── character.py
│   │   └── zeroth_korean
│   │       ├── __init__.py
│   │       └── zeroth_korean.py
│   ├── wav2vec
│   │   ├── __init__.py
│   │   └── trainer.py
├── data
│   ├── kspon_speech
│   │   ├── original
│   │   │   └── ...
│   │   ├── preprocessed
│   │   │   └── transcripts.csv
│   │   │
│   ├── huggingface
│   │   └── ...
├── logs
│   ├── 1
│   │   ├── "run_id"
│   │   │   ├── checkpoints
│   │   │   │   ├── "epoch=xx-cer=xx.ckpt"
└── └── └── └── └── "last.ckpt"
```
