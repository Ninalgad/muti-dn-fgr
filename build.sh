#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t acouslicai_baseline:v0.2.1 \
    -t acouslicai_baseline:latest