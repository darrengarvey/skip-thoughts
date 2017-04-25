#!/usr/bin/env python

import argparse
import json
import numpy as np
import os
import sys
import tensorflow as tf
from annoy import AnnoyIndex
from tensorflow.python.estimator.export.export_lib import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_lib import build_parsing_serving_input_receiver_fn
from tqdm import tqdm
# Local imports
import util
from models import SkipThoughtsModel
from interactive import get_input

"""
This is the process that runs the master, ps and worker nodes in a
distributed setting, or the process that runs everything in a local setup.

If you don't care about tensorboard or just want to KISS, then run this
directly.
"""

def parse_args(args):
  parser = util.get_arg_parser()
  # Add additional command line stuff here...
  parser.add_argument('--num-trees', required=True,
                      help='Number of trees in the index. Higher means more '
                           'accuracy but also more memory (default=%(default)s)')
  parser.add_argument('-i', '--predict-input-pattern', required=True,
                      help='Location to read prediction input data from')
  parser.add_argument('--bidirectional', action='store_true', default=False,
                      help='Use a bidiredctional RNN')
  parser.add_argument('--embedding-dim', default=620, type=int,
                      help='Word embedding dimension (default=%(default)s)')
  parser.add_argument('--encoder-dim', default=2400, type=int,
                      help='Number of units in the RNNCell (default=%(default)s)')
  parser.add_argument('--vocab', required=True, type=str,
                      help='Path to the vocab file used to encode the input')
  parser.add_argument('--batch-size', default=128, type=int,
                      help='Batch size (default=%(default)s)')
  parser.add_argument('--uniform-init-scale', default=0.1, type=float,
                      help='Scale to use for random_uniform_initializer '
                           '(default=%(default)s)')
  return util.parse_args(parser, args)


def main(args):
  args, run_config = parse_args(args)
  model = SkipThoughtsModel(args)
  est = model.get_estimator()
  predictions = est.predict(input_fn=model.get_predict_input,
                            outputs=['thought_vectors'])
  index = AnnoyIndex(args.encoder_dim)
  for i, prediction in tqdm(enumerate(predictions), desc='predictions'):
    #print ('predicted', prediction)
    index.add_item(i, prediction['thought_vectors'])

  index.build(args.num_trees)
  index.save(args.logdir+'/encoded-sentences.ann')


if __name__ == '__main__':
  main(sys.argv[1:])
