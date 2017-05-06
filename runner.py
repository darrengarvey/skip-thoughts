#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
This is the process that runs the master, ps and worker nodes in a
distributed setting, or the process that runs training and evaluation
in a local setup.

If you don't want distributed running or tensorboard then run this
directly.
"""

import argparse
import json
import numpy as np
import os
import sys
import tensorflow as tf
# Local imports
import util
from models import SkipThoughtsModel
from models import SkipThoughtsPolarityClassifierModel
from interactive import get_input

def parse_args(args):
  parser = util.get_arg_parser(__doc__)
  # Add additional command line stuff here...
  group = parser.add_argument_group('Language model arguments')
  group.add_argument('-i', '--input-pattern', required=True,
                      help='Location to read input data from')
  group.add_argument('--pretrained-encoder', type=str,
                      help='Location of frozen graph for a pretrained encoder '
                           'that can output a thought_vectors tensor')
  group.add_argument('-v', '--validation-input-pattern', required=True,
                      help='Location to read validation input data from')
  group.add_argument('--bidirectional', action='store_true', default=False,
                      help='Use a bidiredctional RNN')
  group.add_argument('--embedding-dim', default=620, type=int,
                      help='Word embedding dimension (default=%(default)s)')
  group.add_argument('--encoder-dim', default=2400, type=int,
                      help='Number of units in the RNNCell (default=%(default)s)')
  group.add_argument('--vocab', required=True, type=str,
                      help='Path to the vocab file used to encode the input')
  group.add_argument('--batch-size', default=128, type=int,
                      help='Batch size (default=%(default)s)')
  group.add_argument('--uniform-init-scale', default=0.1, type=float,
                      help='Scale to use for random_uniform_initializer '
                           '(default=%(default)s)')
  group.add_argument('-m', '--model', default='skip',
                      choices=['skip', 'subjectivity'],
                      help='Pick a model')
  return util.parse_args(parser, args)


def main(args):
  args, run_config = parse_args(args)
  model = None
  if args.model == 'skip':
    model = SkipThoughtsModel(args)
  else:
    model = SkipThoughtsPolarityClassifierModel(args)
  util.run(args, run_config, model)
  #serving_input_fn = build_parsing_serving_input_receiver_fn(
  #    model.get_serving_input())
  #model.get_estimator().export_savedmodel(
  #  args.logdir+'/export', serving_input_fn)


if __name__ == '__main__':
  main(sys.argv[1:])
