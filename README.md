
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

