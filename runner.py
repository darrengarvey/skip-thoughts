#!/usr/bin/env python

import argparse
import json
import numpy as np
import os
import sys
import tensorflow as tf
# Local imports
import util
from models import SkipThoughtsModel

"""
This is the process that runs the master, ps and worker nodes in a
distributed setting, or the process that runs everything in a local setup.

If you don't care about tensorboard or just want to KISS, then run this
directly.
"""

def parse_args(args):
  parser = util.get_arg_parser()
  # Add additional command line stuff here...
  parser.add_argument('--input-pattern', required=True,
                      help='Location to read input data from')
  parser.add_argument('--bidirectional', action='store_true', default=False,
                      help='Use a bidiredctional RNN')
  parser.add_argument('--embedding-dim', default=2400, type=int,
                      help='Number of units in the RNNCell (default=%(default)s)')
  parser.add_argument('--vocab', required=True, type=str,
                      help='Path to the vocab file used to encode the input')
  parser.add_argument('--batch-size', default=128, type=int,
                      help='Batch size (default=%(default)s)')
  parser.add_argument('--learning-rate', default=0.002, type=float,
                      help='Learning rate (default=%(default)s)')
  parser.add_argument('--optimizer', default='Adam', type=str,
                      help='Optimizer to use (default=%(default)s)')
  parser.add_argument('--uniform-init-scale', default=0.1, type=float,
                      help='Scale to use for random_uniform_initializer '
                           '(default=%(default)s)')
  return util.parse_args(parser, args)


def main(args):
  args, run_config = parse_args(args)
  model = SkipThoughtsModel(args)
  print ('Task index: %d' % args.task_index)
  util.run(args, model)


if __name__ == '__main__':
  main(sys.argv[1:])
