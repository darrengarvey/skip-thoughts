import argparse
import json
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig


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
  parser.add_argument('--eval-steps', type=int, required=True,
                      help='Number of steps to run when evaluating')
  parser.add_argument('--train-steps', type=int, required=True,
                      help='Number of steps to train for')
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
  else:
    print('Setting up for local training')
    os.environ['TF_CONFIG'] = json.dumps({
      'master': 'grpc://127.0.0.1:2222',
      'cluster': {
        'local': ['localhost:2222'],
      },
      'task': {
        'index': 0,
      }
    })

  run_config = RunConfig()
  return args, run_config


def _create_experiment(args, model):
  """Create a tf.learn.Estimator.

  Args:
      args: Parsed command line from calling `parse_args()`.
      model: An instance of `Model`."""
  def experiment_fn(output_dir):
    estimator = model.get_estimator()
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=model.get_training_input,
        eval_input_fn=model.get_eval_input,
        train_steps=args.train_steps,
        train_monitors=model.get_hooks(),
        eval_steps=args.eval_steps)
  return experiment_fn


def run(args, model):
  """Run a tf.learn.Experiment, possibly distributed."""
  # This is pretty ugly, but we need to set the schedule to this magic
  # string as the default learn_runner figures out is wrong. It's not
  # so easy to fix there either. Meh...
  schedule = 'local_run' if args.environment == 'local' else None
  learn_runner.run(_create_experiment(args, model),
                   args.logdir,
                   schedule=schedule)


class Model(object):
  """Simple abstraction using tf.learn in a possibly distributed
     fashion."""
  def __init__(self, params):
    self.params = params if isinstance(params, dict) else vars(params)
    self.__dict__.update(self.params)

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
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params['learning_rate'],
        optimizer=self.optimizer)

  def get_model(self, features, targets, mode, params):
    """This is the standard model function that needs to be passed to a
       tf.contrib.learn.Estimator(). See:
  
       https://www.tensorflow.org/extend/estimators#constructing_the_model_fn"""
    predictions = self.get_predictions(features, targets, mode, params)
    loss = self.get_loss(predictions, targets, mode, params)
    train_op = self.get_train_op(loss, params)
    eval_metric_ops = self.get_eval_metrics(features, predictions, targets,
                                            mode, params)
    return tf.contrib.learn.ModelFnOps(mode, predictions, loss, train_op,
                                       eval_metric_ops)
 
  def get_estimator(self):
    """By default, we used the default estimator, but users can override
       this if they like."""
    return tf.contrib.learn.Estimator(model_fn=self.get_model,
                                      model_dir=self.params['logdir'],
                                      params=self.params)

