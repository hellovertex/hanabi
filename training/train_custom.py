# coding=utf-8
#
"""The entry point for running a Rainbow agent on Hanabi.

example call from base directory (hanabi):
python -m training.train_custom
--gin_files training/rainbow_training/configs/default_experiment_rainbow.gin
--base_dir training/rainbow_training
--checkpoint_dir checkpoints
--logging_dir logs

importantly, don't end filepaths with '/'

Provides information from commandline arguments and gin config to configure experiment with
functionality from run_experiment.py. The environment is created using rl_env.make() and the experiment then started
with run_experiment.run_experiment()

Configuration of the experiment is handled in the following way:
The configuration for the GAME DYNAMICS are stored in the dict configs which gets used by HanabiGame, HanabiEnv,
RewardMetrics and ObservationAugmenter at several points:
rl_env.make chooses a pre-set config dictionary depending on the environment name (specified in gin file) containing
    colors
    ranks
    players
    hand_size
    max_information_tokens
    max_life_tokens
    observation_type
Let's call these the default set of configuration parameters included in config.
rl_env.HanabiEnv also uses (but doesn't seem to require)
    seed
    random_start_player
in its init, config is also passed to
 -  pyhanabi.HanabiGame() where config is called params and the two new params are apparently set to default values
    inside hanabi_lib.NewGame
 -  RewardMetrics() that also requires the following configs corresponding to the custom reward scheme
        per_card_reward
        _custom_reward=.2
        _penalty_last_hint_token_used=.2
 -  ObservationAugmenter() which doesn't require additional configuration settings

However, the rl_env.make's environment name is a gin configurable, so the gin configs sort of wrap at least the
default set of configurations as far as a corresponding dict is defined to a environment name in rl_env.make.

The configuration for the AGENT on the other hand are set as gin.configs in the file provided by the parsed command
line arguments. This includes in the case of the Rainbow agent
 -  the template, i.e. the blueprint of layers
 -  RainbowAgents' hyperparameters regarding the learning
as well as e.g. the ObservationStacker's depth.

The most convenient way to provide parameters to RewardMetrics (ObservationAugmenter doesn't need any) is to make
these gin.configurable too.
Two problems:
    using gin we can't append per-card_reward, _custom_reward and _penalty_last_hint_token_used to the dict
        (only overwrite config dict). for now, this is solved by adding a default value True for per_card_reward in
        the initialization of RewardMetrics
    the flags to use custom rewards are set as global variables at the beginning of rl_env. idea: make function
        handle it, make this function gin configurable

Sascha's train_ppo_custom_env.py sets the config dict from a lists of vals as global vars in the beginning,
looping over all combinations to construct a specific config. This config is then used to create environments etc.
etc.

Where do the print statements and logs come from?
in run experiment, after every iteration run_experiment logs at INFO level
    -how long the iteration took
    -how long the logging took
    -how long the checkpointing took
and at the same time run_one_iteration logs at INFO level
    -average training steps per second
    -average per episode return
If the current step is an evaluation step, this function also logs at INFO level
    -average eval episode length
    -average eval return
run_one_episode also logs at DEBUG level
    -current episode length
    -current episode return



todo: okay basically seems to work,
    how many iterations are necessary?
    script to load trained agent -> best try it out by loading two checkpoints, let them play a round against each other
    does agent contain hyperparameters, like configs?
    activate additional rewards, see whether it works
        find print statements in custom reward scheme and make them more informative,
        train Rainbow seer/private/6 combinations of rewards, DQN private 6 combinations of rewards
    write script to train rewards consecutively
    make 'per_card_reward', '_custom_reward' and '_penalty_last_hint_token_used't gin configurables, too
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

# import hanabi_learning_environment.agents.rainbow
import sys
sys.path.append("./hanabi_learning_environment/agents/rainbow")

from third_party.dopamine import logger
import run_experiment

# from hanabi_learning_environment.agents.rainbow.third_party.dopamine import logger
# from hanabi_learning_environment.agents.rainbow import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

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


def launch_experiment():
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir == None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

  environment = run_experiment.create_environment()
  obs_stacker = run_experiment.create_obs_stacker(environment)
  agent = run_experiment.create_agent(environment, obs_stacker)

  checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
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


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)
