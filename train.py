#!/usr/bin/env python

import argparse
import itertools
import sys
from six.moves import shlex_quote


def parse_args(args):
  parser = argparse.ArgumentParser(description='Spawn processes and run training.')
  parser.add_argument('-s', '--session', default='skip-thoughts',
                      help='Use this name to identify the running session. Only '
                           'useful if running this multiple times simultaneously.')
  parser.add_argument('-w', '--num-workers', default=1, type=int,
                      help='Number of workers.')
  parser.add_argument('--num-ps', default=1, type=int,
                     help='Number of parameter nodes (default of 1 is fine for local runs).')
  parser.add_argument('-p', '--starting_port', default=23456,
                     help='Number of parameter nodes.')
  parser.add_argument('-l', '--logdir', type=str, required=True,
                      help='Log directory path.')
  parser.add_argument('-n', '--dry-run', action='store_true',
                      help='Print out commands rather than executing them.')
  parser.add_argument('-m', '--mode', default='tmux',
                      choices=['tmux', 'nohup', 'child'],
                      help='tmux: run workers in a tmux session. '
                           'nohup: run workers with nohup. '
                           'child: run workers as child processes')
  parser.add_argument('-g', '--use-gpus', type=str, default='0',
                     help='Use these GPU(s) if available (eg. "0,1,2,3", or "" (ie. none)')
  return parser.parse_args(args)


class JobSpawner(object):
  def __init__(self, args):
    self.logdir = args.logdir
    self.session = args.session
    self.mode = args.mode
    self.num_ps = args.num_ps
    self.num_workers = args.num_workers
    self.tensorboard_port = args.starting_port
    self.port = args.starting_port + 1
    gpus = args.use_gpus.split(',')
    self.gpus = itertools.cycle(gpus if len(gpus) else [''])
    # The list of commands we'll want to run
    self.cmds = []
    self.cmds_map = {}
    # TODO: These could be configurable I guess...
    self.runner_script = 'runner.py'
    self.host = '127.0.0.1'
    self.shell = 'bash'

    self._create_cluster_spec()
    self._create_commands()

  def _create_cluster_spec(self):
    # The order of things in this function matter!
    cluster = {
      'master': '{}:{}'.format(self.host, self.port),
      'ps': [],
      'worker': [],
    }
    for _ in range(self.num_ps):
      self.port += 1
      cluster['ps'].append('{}:{}'.format(self.host, self.port))
    for _ in range(self.num_workers):
      self.port += 1
      cluster['worker'].append('{}:{}'.format(self.host, self.port))
    self.cluster = cluster

  def _new_cmd(self, name, cmd):
    if isinstance(cmd, (list, tuple)):
      cmd = ' '.join(shlex_quote(str(v)) for v in cmd)
    if self.mode == 'tmux':
      cmd = 'tmux send-keys -t {}:{} {} Enter'.format(
          self.session, name, shlex_quote(cmd))
    elif self.mode == 'child':
      cmd = '{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh'.format(
          cmd, self.logdir, self.session, name, self.logdir)
    elif self.mode == 'nohup':
      cmd = 'nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh'.format(
          self.shell, shlex_quote(cmd), self.logdir, self.session, name, self.logdir)
    self.cmds_map[name] = cmd
  
  def _create_commands(self):
    """Get a list of commands to run to invoke:
       - master server
       - parameter server(s)
       - worker node(s)
       - tensorboard"""
    base_cmd = [
        sys.executable, self.runner_script,
        '--logdir', self.logdir,
        '--master', self.cluster['master'],
        '--ps-hosts', ','.join(self.cluster['ps']),
        '--worker-hosts', ','.join(self.cluster['worker']),
    ]
  
    # None of these need the GPU.
    self._new_cmd(
        'tb', ['tensorboard', '--logdir', self.logdir, '--port', self.tensorboard_port])
    if self.mode == 'tmux':
      self._new_cmd('htop', ['htop'])
  
    # Keep track of the current task index into the cluster spec.
    index = 0
    self._new_cmd('master',
        ['CUDA_VISIBLE_DEVICES='] + base_cmd + ['--task-name', 'master',
                                                '--task-index', index])
    for i in range(self.num_ps):
      index += 1
      self._new_cmd('ps',
          ['CUDA_VISIBLE_DEVICES='] + base_cmd + ['--task-name', 'ps',
                                                  '--task-index', index])

    # Now set up the workers with GPU access. 
    for i in range(self.num_workers):
      index += 1
      self._new_cmd('w-%d' % i,
          ['CUDA_VISIBLE_DEVICES={}'.format(self.gpus.next())] + base_cmd +
          ['--task-name', 'worker', '--task-index', index])
  
    notes = []
    cmds = [
        'mkdir -p {}'.format(self.logdir),
        'echo {} {} > {}/cmd.sh'.format(
            sys.executable,
            ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']),
            self.logdir),
    ]
    if self.mode in ['nohup', 'child']:
      cmds += ['echo "#!/bin/{}" >{}/kill.sh'.format(self.shell, self.logdir)]
      notes += ['Run `source {}/kill.sh` to kill the job'.format(self.logdir)]
    if self.mode == 'tmux':
      notes += ['Use `tmux attach -t {}` to watch process output'.format(self.session)]
      notes += ['Use `tmux kill-session -t {}` to kill the job'.format(self.session)]
    else:
      notes += ['Use `tail -f {}/*.out` to watch process output'.format(self.logdir)]
    notes += ['Point your browser to http://{}:{} to see Tensorboard'.format(
        self.host, self.tensorboard_port)]
  
    windows = self.cmds_map.keys()
  
    if self.mode == 'tmux':
      cmds += [
          'tmux kill-session -t {}'.format(self.session),
          'tmux new-session -s {} -n {} -d {}'.format(
              self.session, windows[0], self.shell)
      ]
      for w in windows[1:]:
        cmds += ['tmux new-window -t {} -n {} {}'.format(self.session, w, self.shell)]
      cmds += ['sleep 1']
    cmds += self.cmds_map.values()
  
    self.cmds = cmds
    self.notes = notes

  def print_commands(self):
    print("\n".join(self.cmds))

  def print_notes(self):
    print("\n".join(self.notes))

  def run(self):
    return os.system('\n'.join(self.cmds))


def main(args):
  args = parse_args(args)
  if args.dry_run:
    print('Dry-run mode due to -n flag, otherwise the following '
          'commands would be executed:')
  else:
    print('Executing the following commands:')
  runner = JobSpawner(args)
  runner.print_commands()
  if not args.dry_run:
    if args.mode == 'tmux':
      os.environ['TMUX'] = ''
    runner.run()
  runner.print_notes()


if __name__ == '__main__':
  main(sys.argv[1:])
