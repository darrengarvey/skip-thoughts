#!/bin/bash

this_dir=$(dirname $(readlink -f "${BASH_SOURCE}"))
data_dir="/data/sources/dretzq/equal-news/data/skip-thought-small-60"

# lr=0.002
# lr=0.001 @ step 14k
# lr=0.0005 @ step 18251
# lr=0.00025 @ step 37473
# lr=0.0005 @ step ~40k, but incl. exponential decay
# lr=0.0003+exp @ step 41949
export CUDA_VISIBLE_DEVICES=1
step=${1:-70000}
while true; do
    time /usr/bin/python runner.py \
        --logdir ${this_dir} \
        --learning-rate 0.0005 \
        --input-pattern "${data_dir}/train-*" \
        --validation-input-pattern ${data_dir}/validation-00000-of-00001 \
        --train-steps ${step} \
        --eval-steps 100 \
        --vocab ${data_dir}/vocab.txt \
        --debug
    step=$(($step + 5000))
done
