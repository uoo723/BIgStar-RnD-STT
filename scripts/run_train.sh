#!/usr/bin/env bash
set -e

./scripts/run_wav2vec_kspon.sh 0 && \
RUN_ID=$(cat ./run_id) ./scripts/run_wav2vec_zeroth.sh 0
