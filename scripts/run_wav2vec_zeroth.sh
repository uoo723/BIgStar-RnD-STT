#!/usr/bin/env bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=Wav2Vec

DATASET=zeroth_korean
MODEL=Wav2Vec
RUN_ID=${RUN_ID:-05ec8c66943644639b01e52b03a0fae3}

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --optim-name "adamw"
    --lr 3e-4
    --ctc-loss-reduction "sum"
    --ctc-zero-infinity
    --num-epochs 100
    --train-batch-size 4
    --test-batch-size 4
    --accumulation-step 4
    --scheduler-type "linear"
    --scheduler-warmup 0.03
    --early-criterion 'cer'
    --seed $1
    --swa-warmup 0
    --eval-step 2000
    --early 20
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "Wav2Vec"
    --valid-size 1000
    --run-id $RUN_ID
    --load-only-weights
    --load-best
)

python main.py train-wav2vec "${args[@]}"
