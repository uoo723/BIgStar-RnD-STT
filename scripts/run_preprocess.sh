#!/usr/bin/env bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python main.py preprocess-kspon
