#!/bin/bash

# containing model.ckpt*
LOGDIR="$1"
# the step to save (needs a model.chkp-NUM file)
CHECKPOINT_NUM="$2"
# The variable we want to mark as the output
OUTPUT_NODE="thought_vectors"

TF_DIR=/data/software/machine-learning/tensorflow/bazel-bin/

# Don't bother using up the GPU for this. It doesn't help.
export CUDA_VISIBLE_DEVICES=

${TF_DIR}/tensorflow/python/tools/freeze_graph \
    --input_graph ${LOGDIR}/graph.pbtxt \
    --input_checkpoint ${LOGDIR}/model.ckpt-${CHECKPOINT_NUM} \
    --output_graph ${LOGDIR}/graph-${CHECKPOINT_NUM}.pb \
    --output_node_names ${OUTPUT_NODE}
