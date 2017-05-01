import argparse
import json
import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig
from tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils import make_export_strategy
from tensorflow.python.estimator.export.export_lib import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_lib import build_parsing_serving_input_receiver_fn

"""
Code that you shouldn't have to care about too much. If tf.learn changes
it's interface (which it always does), this should be the only place to
have to change.

Also makes working in local / distributed uniform.
"""

def get_arg_parser():
  parser = argparse.ArgumentParser(description='Start a distributed job')
  parser.add_argument('--debug', default=False, action="store_true",
                      help='Turn on debug logging')
  ###################################################################
  # Distributed running arguments
  parser.add_argument('--environment', default='local',
                      choices=['cloud', 'local'],
                      metavar=['cloud', 'local'],
                      help='Whether to run locally or in a cloud setting.')
  parser.add_argument('-t', '--task-index', default=0, type=int,
                      help='Worker task index. Task index=0 is the master'
                           'worker, so does initialisation')
  parser.add_argument('--master', help='host of the master')
  parser.add_argument('--task-name', default='none',
                      choices=['master', 'worker', 'ps', 'none'],
                      metavar=['master', 'worker', 'ps', 'none'],
                      help='The job type')
  parser.add_argument('--ps-hosts', default='',
                      help='Comma-separated list of hostname:port pairs')
  parser.add_argument('--worker-hosts', default='',
                      help='Comma-separated list of hostname:port pairs')
  ###################################################################
  parser.add_argument('--logdir', required=True,
                      help='Location to store checkpoint / log data')
  parser.add_argument('--optimizer', default='Adam', type=str,
                      help='Optimizer to use (default=%(default)s)')
  parser.add_argument('--jit', action='store_true', default=False,
                      help='Use XLA to compile the graph (experimental)')
  parser.add_argument('--eval-steps', type=int,
                      help='Number of steps to run when evaluating')
  parser.add_argument('--min_eval_frequency', default=1000, type=int,
                      help='Min number of steps between eval steps. '
                     'Evaluation runs on the CPU (if you use train.py) '
                     'and the GPU can get starved if the CPU is busy '
                     'running evaluation')
  parser.add_argument('--train-steps', type=int,
                      help='Number of steps to train for')
  parser.add_argument('--clip-gradients', default=5.0, type=float,
                      help='Clip gradients over this value (default=%(default)s)')
  parser.add_argument('--learning-rate', default=0.002, type=float,
                      help='Learning rate (default=%(default)s)')
  parser.add_argument('--learning-rate-decay-rate', default=0.5, type=float,
                      help='Learning rate decay control (default=%(default)s)')
  parser.add_argument('--learning-rate-decay-steps', default=50000, type=int,
                      help='How often to decay the learning rate '
                           '(default=%(default)s)')
  return parser


def parse_args(parser, args):
  """Parse args and set up logging and TF_CONFIG for distributed
     training."""
  args = parser.parse_args(args)
  # Turn up logging to get a better idea what's going on.
  if args.debug:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  else:
    tf.logging.set_verbosity(tf.logging.INFO)

  if args.ps_hosts != '':
    print('Setting up for distributed training')
    os.environ['TF_CONFIG'] = json.dumps({
      'master': args.master,
      'environment': args.environment,
      'cluster': {
        'master': [args.master],
        'ps': args.ps_hosts.split(','),
        'worker': args.worker_hosts.split(','),
      },
      'task': {
        'type': args.task_name,
        'index': args.task_index,
      }
    })

  run_config = RunConfig(model_dir=args.logdir)
  return args, run_config


def _create_experiment(args, model):
  """Create a tf.learn.Estimator.

  Args:
      args: Parsed command line from calling `parse_args()`.
      model: An instance of `Model`."""
  def experiment_fn(run_config, hparams):
    estimator = model.get_estimator()
    if hasattr(model, 'get_serving_input'):
      # What an ugly function name...
      print ('Setting up for export')
      serving_input_fn = build_raw_serving_input_receiver_fn(
          model.get_serving_input)
      export_strategy = make_export_strategy(serving_input_fn)
    else:
      export_strategy = None
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=model.get_training_input,
        eval_input_fn=model.get_eval_input,
        train_steps=args.train_steps,
        train_monitors=model.get_hooks(),
        eval_steps=args.eval_steps,
        min_eval_frequency=args.min_eval_frequency)
  return experiment_fn


def run(args, run_config, model):
  """Run a tf.learn.Experiment, possibly distributed."""
  if args.jit:
      jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
      with jit_scope():
          _run(args, run_config, model)
  else:
      _run(args, run_config, model)

def _run(args, run_config, model):
  # This is pretty ugly, but we need to set the schedule to this magic
  # string as the default learn_runner figures out is wrong. It's not
  # so easy to fix there either. Meh...
  schedule = None
  learn_runner.run(_create_experiment(args, model),
                   run_config=run_config,
                   schedule=schedule)


class Model(object):
  """Simple abstraction using tf.learn in a possibly distributed
     fashion."""
  def __init__(self, params):
    self.params = params if isinstance(params, dict) else vars(params)
    self.__dict__.update(self.params)
    self._estimator = None


  def get_training_input(self):
    """Get input for training runs.
  
       Return a function that will return two dictionaries. One of input
       tensors and another of output tensors.
  
       These tensors are used by the Estimator based on the
       FeatureColumns you pass into it."""
    return None

  def get_eval_input(self):
    """Get input for evaluation runs.
  
       Return a function that will return two dictionaries. One of input
       tensors and another of output tensors.
  
       These tensors are used by the Estimator based on the
       FeatureColumns you pass into it."""
    return None

  def get_hooks(self):
    """List of additional SessionRunHooks to use while training."""
    return None

  def get_eval_metrics(self, features, predictions, targets, mode, params):
    """Return a dictionary of metrics to collect when evaluating.
       See tf.metrics for some examples."""
    return None

  def get_predictions(self, features, targets, mode, params):
    """This is your model. Return the output of your model."""
    raise NotImplementedError("get_predictions not implemented")
 
  def get_loss(self, predictions, targets, mode, params):
    """This is the loss for your model. The predictions are the output of
       get_predictions(). If you need more control instead implement
       get_model()."""
    raise NotImplementedError("get_loss not implemented")

  def get_train_op(self, loss, params):
    learning_rate_decay_fn = None
    if self.learning_rate_decay_rate > 0:
      def learning_rate_decay(lr, global_step):
        return tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=global_step,
            decay_rate=self.learning_rate_decay_rate,
            decay_steps=self.learning_rate_decay_steps,
            staircase=False)
      learning_rate_decay_fn = learning_rate_decay
    else:
      learning_rate_decay_fn = None
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        clip_gradients=self.clip_gradients,
        learning_rate=self.learning_rate,
        learning_rate_decay_fn=learning_rate_decay_fn,
        summaries=optimizers.OPTIMIZER_SUMMARIES,
        optimizer=self.optimizer)

  def get_model(self, features, labels, mode, params):
    """This is the standard model function that needs to be passed to a
       tf.contrib.learn.Estimator(). See:
  
       https://www.tensorflow.org/extend/estimators#constructing_the_model_fn"""
    predictions = self.get_predictions(features, labels, mode, params)
    if mode == tf.contrib.learn.ModeKeys.INFER:
      loss = None
      train_op = None
      eval_metric_ops = None
    else:
      loss = self.get_loss(predictions, labels, mode, params)
      train_op = self.get_train_op(loss, params)
      eval_metric_ops = self.get_eval_metrics(features, predictions, labels,
                                              mode, params)
    return tf.contrib.learn.ModelFnOps(mode, predictions, loss, train_op,
                                       eval_metric_ops)
 
  def get_estimator(self):
    """By default, we used the default estimator, but users can override
       this if they like."""
    if not self._estimator:
      def model_fn(features, labels, mode, params):
        return self.get_model(features, labels, mode, params)
      self._estimator = tf.contrib.learn.Estimator(
          model_fn=model_fn, model_dir=self.params['logdir'],
          params=self.params)
    return self._estimator

