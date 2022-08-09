#!/usr/bin/env bash
set -e

# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=Wav2Vec

DATASET=zeroth_korean
MODEL=Wav2Vec

if [[ -f "./run_id_2" ]]; then
    RUN_ID=$(cat ./run_id_2)
fi

RUN_ID=${RUN_ID:-368a860ddbb04eeca7558da8e84f88f9}

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --mode "test"
    --test-batch-size 4
    --mp-enabled
    --num-workers 8
    --experiment-name "Wav2Vec"
    --run-id $RUN_ID
)

python main.py train-wav2vec "${args[@]}"
