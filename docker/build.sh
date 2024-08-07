#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" --no-cache \
    -t backbone:v0.2.1 \
    -t backbone:latest