# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""The entry point for training a Rainbow agent on Hanabi.
python3 -um train_rainbow_variants --base_dir="{base_dir}" --gin_files="{gin_file}"
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

from agents.rainbow_copy.third_party.dopamine import logger
import agents.run_experiment as run_experiment


flags.DEFINE_multi_string(
    'gin_files', [str(os.path.dirname(__file__)) + '/configs/hanabi_rainbow.gin'],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', str(os.path.dirname(__file__)) + '/logs/Rainbow',
                    'Base directory to host all required sub-directories. '
                    'Path for logs and checkpoints')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')
flags.DEFINE_string('game_type', 'Hanabi-Full',
                    'Hanabi-Full or Hanabi-Small, etc.')
FLAGS = flags.FLAGS


def main(unused_argv):
    """ Runs the experiment. """

    # Load the gin configs and bindings.
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    # Initialize the Logger object.
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))
    # Initialize the environment.
    environment = run_experiment.create_environment(game_type='Hanabi-Small')
    # Initialize the observation stacker.
    obs_stacker = run_experiment.create_obs_stacker(environment)
    # Initialize the agent.
    agent = run_experiment.create_agent(environment, obs_stacker, agent_type='Rainbow')
    # Reload latest checkpoint, if available, and initialize Checkpointer object
    checkpoint_dir = f'{FLAGS.base_dir}/checkpoints'
    start_iteration, experiment_checkpointer = (
        run_experiment.initialize_checkpointing(agent,
                                                experiment_logger,
                                                checkpoint_dir,
                                                FLAGS.checkpoint_file_prefix))

    run_experiment.run_experiment(agent, environment, start_iteration,
                                  obs_stacker,
                                  experiment_logger, experiment_checkpointer,
                                  checkpoint_dir,
                                  logging_file_prefix=FLAGS.logging_file_prefix)


if __name__ == '__main__':
    # flags.mark_flag_as_required('base_dir')
    # flags.mark_flag_as_required('gin_files')
    app.run(main)
