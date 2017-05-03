#!/bin/bash

input="/data/sources/dretzq/equal-news/data/plain-small/*.txt"
output="$(dirname $(readlink -f ${BASH_SOURCE}))"
# Max number of sentences (this default includes all ~4.6m of them)
max="${1:-5000000}"

time ./bazel-bin/skip_thoughts/data/preprocess_dataset \
    --vocab_file vocab.txt \
    --input_files "$input" \
    --output_dir "$output" \
    --add_eos \
    --max_sentences "$max" \
    --max_sentence_length 60 \
    --train_output_shards 15
