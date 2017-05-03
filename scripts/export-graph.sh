#!/bin/bash

# containing model.ckpt*
LOGDIR="$1"
# the step to save (needs a model.chkp-NUM file)
CHECKPOINT_NUM="$2"

this_dir="$(dirname ${BASH_SOURCE})"

${this_dir}/freeze-graph.sh "$LOGDIR" $CHECKPOINT_NUM
${this_dir}/optimise-graph.sh "$LOGDIR" $CHECKPOINT_NUM
