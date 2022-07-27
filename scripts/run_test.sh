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
    --mode "test"
    --test-batch-size 16
    --mp-enabled
    --num-workers 0
    --experiment-name "Wav2Vec"
    --run-id "f8ff3c51e29f4768830cd1ec02f785fc"
)

python main.py train-wav2vec "${args[@]}"
