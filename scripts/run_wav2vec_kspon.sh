#!/usr/bin/env bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=Wav2Vec

DATASET=kspon
MODEL=Wav2Vec

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --optim-name "adamw"
    --num-hidden-layers 6
    --num-attention-heads 4
    --intermediate-size 1024
    --lr 5e-5
    --ctc-loss-reduction "sum"
    --ctc-zero-infinity
    --num-epochs 30
    --train-batch-size 4
    --test-batch-size 4
    --accumulation-step 4
    --scheduler-type "linear"
    --scheduler-warmup 0.02
    --early-criterion 'cer'
    --seed $1
    --swa-warmup 0
    --eval-step 5000
    --early 20
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "Wav2Vec"
    --valid-size 1000
    --skip-test
    --save-run-id-path "./run_id"
)

python main.py train-wav2vec "${args[@]}"
