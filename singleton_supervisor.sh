#!/usr/bin/env bash

contexts="0 1 2 4 16 1600"
for context in $contexts
do
for tr in $(seq 500 1295)
do
bash ./lib/run_singleton_embedding.sh $context $tr
done
done