Examples
--------

Run on single machine:

	/usr/bin/python runner.py \
	    --logdir ${this_dir} \
	    --learning-rate 0.00025 \
	    --input-pattern "${data_dir}/train-*" \
	    --validation-input-pattern ${data_dir}/validation-00000-of-00001 \
	    --train-steps 100000 \
	    --eval-steps 500 \
	    --vocab ${data_dir}/vocab.txt \
	    --debug

Run with multiple GPUs on a single machine:

    ./train.py -n \
        --num-workers 2 \
        --gpus 0,1 \
        --logdir logdir/take-2 \
        --eval-steps 1000 \
        --train-steps 10000 \
        --input-pattern "data/skip-thought/train-*"

Freeze graph
------------

    LOGDIR=/path/to/logdir  # containing model.ckpt*
    CHECKPOINT_NUM=12345    # the step to save (needs a model.chkp-NUM file)
    CUDA_VISIBLE_DEVICES= ./bazel-bin/tensorflow/python/tools/freeze_graph \
        --input_graph ${LOGDIR}/graph.pbtxt \
        --input_checkpoint ${LOGDIR}/model.ckpt-${CHECKPOINT_NUM} \
        --output_graph ${LOGDIR}/graph-${CHECKPOINT_NUM}.pb \
        --output_node_names thought_vectors

Optimize the graph
------------------

    LOGDIR=/path/to/logdir  # containing model.ckpt*
    CHECKPOINT_NUM=12345    # the step to save (needs a model.chkp-NUM file)
    CUDA_VISIBLE_DEVICES= ./bazel-bin/tensorflow/python/tools/optimize_for_inference \
        --input ${LOGDIR}/graph-${CHECKPOINT_NUM}.pb \
        --output ${LOGDIR}/graph-${CHECKPOINT_NUM}.optimized.pb \
        --output_names thought_vectors
