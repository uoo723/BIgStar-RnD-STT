#!/usr/bin/env bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=Wav2Vec

DATASET=dataset
MODEL=Wav2Vec

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --optim-name "adamw"
    --lr 2.5e-4
    --num-epochs 30
    --train-batch-size 2
    --test-batch-size 4
    --accumulation-step 4
    --scheduler-type "linear"
    --scheduler-warmup 500
    --early-criterion 'cer'
    --seed $1
    --swa-warmup 1
    --eval-step 5000
    --early 50
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "Wav2Vec"
    --valid-size 300
)

python main.py train-wav2vec "${args[@]}"
