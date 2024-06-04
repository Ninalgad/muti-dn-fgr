#!/usr/bin/env bash

./build.sh

docker save acouslicai_baseline | gzip -c > acouslicai_baseline.tar.gz