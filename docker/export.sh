#!/usr/bin/env bash

./build.sh

docker save backbone | gzip -c > backbone.tar.gz
