from absl import app, flags
import agents.run_experiment as run_experiment
import hanabi_learning_environment.rl_env as rl_env
from agents.rainbow_copy.third_party.dopamine import logger
from agents.rainbow_copy import rainbow_agent
import os

flags.DEFINE_multi_string(
    'gin_files', [str(os.path.dirname(__file__)) + '/configs/hanabi_rainbow.gin'],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('game_type', 'Hanabi-Small',
                    'One of {Hanabi-Full, Hanabi-Small, ...} c.f. hanabi-learning-environment')
flags.DEFINE_integer('num_players', 4,
                     'Number of players')
flags.DEFINE_string('root_dir', str(os.path.dirname(__file__)) + '/logs/Rainbow',
                    'Base directory to host all required sub-directories. '
                    'Path for logs and checkpoints')
FLAGS = flags.FLAGS


def main(_):
    experiment_logger = logger.Logger(f'{FLAGS.root_dir}')
    env = rl_env.make(environment_name=FLAGS.game_type, num_players=FLAGS.num_players)
    obs_stacker = run_experiment.create_obs_stacker(environment=env, history_size=1)
    agent = rainbow_agent.RainbowAgent(
        observation_size=obs_stacker.observation_size(),
        num_actions=env.num_moves(),
        num_players=env.players)

    # reload checkpoint if possible
    start_iter, checkpointer = run_experiment.initialize_checkpointing(
        agent=agent,
        experiment_logger=experiment_logger,
        checkpoint_dir=f'{FLAGS.base_dir}/checkpoints'
    )
    run_experiment.run_experiment(
        agent=agent,
        environment=env,
        start_iteration=start_iter,
        obs_stacker=obs_stacker,
        experiment_logger=experiment_logger,
        experiment_checkpointer=checkpointer,
        checkpoint_dir=f'{FLAGS.base_dir}/checkpoints',
        summary_dir=f'{FLAGS.base_dir}/summary',

    )


if __name__ == 'main':
    # flags.mark_flag_as_required('root_dir')
    app.run(main)
