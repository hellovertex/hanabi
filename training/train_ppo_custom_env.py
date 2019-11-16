# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval PPO.

To run:
EXAMPLE USAGE
```python3 -um train_ppo_custom_env --root_dir '/home/hellovertex/Documents/tensorboards/Sascha/ppo/custom_envs/'
--summary_dir '/home/hellovertex/Dropbox/Hanabi\ Learning\ Environment\ etc.../'

summary_dir stores only the tfevent file for tensorboard
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import batched_py_metric
from tf_agents.metrics import tf_metrics
from tf_agents.metrics.py_metrics import AverageEpisodeLengthMetric
from tf_agents.metrics.py_metrics import AverageReturnMetric
from tf_agents.policies import py_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

# own imports
from custom_environment.custom_metric import PyScoreMetric, TfScoreMetric

sys.path.insert(0, 'lib')
from hanabi_learning_environment import rl_env
from training.tf_agents_lib import masked_networks, pyhanabi_env_wrapper

flags.DEFINE_string('root_dir', str(os.path.dirname(__file__)) + '/logs/hanabi_small/ppo/',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('summary_dir', str(os.path.dirname(__file__)) + '/summaries/hanabi_small/ppo/',
                    'Directory for writing tensorboards summaries.')
flags.DEFINE_string('master', '', 'master session')
flags.DEFINE_integer('replay_buffer_capacity', 1001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 40,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_environment_steps', int(3e08),
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 30,
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('num_eval_episodes', 30,
                     'The number of episodes to run eval on.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
FLAGS = flags.FLAGS


COLORS = [2]
RANKS = [5]
NUM_PLAYERS = [2, 4]
HAND_SIZES = [2]
MAX_INFORMATION_TOKENS = [3]
# MAX_LIFE_TOKENS = [2,3]
MAX_LIFE_TOKENS = [1]
OBSERVATION_TYPE = 1  # pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
CUSTOM_REWARDS = [.2]
PENALTIES_LAST_HINT_TOKEN = [.1, .5]


def load_hanabi_env(game_config):
    assert isinstance(game_config, dict)

    pyhanabi_env = rl_env.HanabiEnv(game_config)

    if pyhanabi_env is not None:
        return pyhanabi_env_wrapper.PyhanabiEnvWrapper(pyhanabi_env)

    return None


def format_dir(dir, game_config):
    """ Outputs directory for tensorboard summary writing, corresponding to current game_config
    Example: dir/2_players/hand_size=2/max_life_tokens=2/colors=2,ranks=2/max_info_tokens=3,ot=1/
    """
    first_lvl = f'{game_config["players"]}_players'
    second_lvl = f'hand_size={game_config["hand_size"]}'
    third_lvl = f'max_life_tokens={game_config["max_life_tokens"]}'
    fourth_lvl = f'colors={game_config["colors"]},ranks={game_config["ranks"]}'
    fifth_lvl = f'max_info_tokens={game_config["max_information_tokens"]},ot={game_config["observation_type"]}'

    tmp = os.path.join(dir, first_lvl)
    tmp = os.path.join(tmp, second_lvl)
    tmp = os.path.join(tmp, third_lvl)
    tmp = os.path.join(tmp, fourth_lvl)
    formatted_summary_dir = os.path.join(tmp, fifth_lvl)
    if 'custom_reward' in game_config:
        custom_reward = f'custom_reward={game_config["custom_reward"]}'
        formatted_summary_dir = os.path.join(formatted_summary_dir, custom_reward)
    if 'penalty_last_hint_token' in game_config:
        penalty = f'penalty={game_config["penalty_last_hint_token"]}'
        formatted_summary_dir = os.path.join(formatted_summary_dir, penalty)

    return formatted_summary_dir


def get_networks(tf_env, networks_layers):
    assert isinstance(networks_layers, dict)
    actor_fc_layers = networks_layers["actor_net"]
    value_fc_layers = networks_layers["value_net"]
    actor_net = masked_networks.MaskedActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=actor_fc_layers)
    value_net = masked_networks.MaskedValueNetwork(
        tf_env.observation_spec(), fc_layer_params=value_fc_layers)
    return actor_net, value_net


def get_rnn_networks(tf_env, network_layers, lstm_size=(75,75)):
    """ Question is, whether we should use RNN for value net as well, yes so? """
    raise NotImplementedError


def get_metrics_eval(num_parallel_environments, num_eval_episodes):
    eval_metrics = [
        batched_py_metric.BatchedPyMetric(
            AverageReturnMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=num_parallel_environments),
        batched_py_metric.BatchedPyMetric(
            AverageEpisodeLengthMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=num_parallel_environments),
        batched_py_metric.BatchedPyMetric(
            PyScoreMetric,
            metric_args={'buffer_size': num_eval_episodes},
            batch_size=num_parallel_environments)
    ]
    return eval_metrics


def get_metrics_train_and_step(num_eval_episodes, num_parallel_environments):
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    environment_steps_count = environment_steps_metric.result()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
        TfScoreMetric()
    ]

    return train_metrics, step_metrics, environment_steps_count


def get_writers_train_eval(summary_dir, eval_dir, game_config, summaries_flush_secs=1):
    formatted_summary_dir = format_dir(summary_dir, game_config)

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        formatted_summary_dir, flush_millis=summaries_flush_secs * 1000)  #
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    return train_summary_writer, eval_summary_writer


def train_eval(
        root_dir,
        summary_dir,
        game_config,
        tf_master='',
        env_load_fn=None,
        random_seed=0,
        # TODO(b/127576522): rename to policy_fc_layers.
        actor_fc_layers=(150, 75),
        value_fc_layers=(150, 75),
        actor_fc_layers_rnn=(150,),
        value_fc_layers_rnn=(150,),
        use_rnns=False,
        # Params for collect
        num_environment_steps=int(3e06),
        collect_episodes_per_iteration=90,
        num_parallel_environments=30,
        replay_buffer_capacity=1001,  # Per-environment
        # Params for train
        num_epochs=25,
        learning_rate=1e-4,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=500,
        # Params for summaries and logging
        train_checkpoint_interval=2000,
        policy_checkpoint_interval=1000,
        rb_checkpoint_interval=4000,
        log_interval=500,
        summary_interval=500,
        summaries_flush_secs=1,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None,
        eval_py_env=None,
        tf_env=None
        ):
    tf.reset_default_graph()
    """A simple train and eval for PPO."""
    if root_dir is None:
        raise AttributeError('train_eval requires a root_dir.')

    # ################################################ #
    # ------------ Create summary-writers ------------ #
    # ################################################ #
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(format_dir(root_dir, game_config), 'train')
    eval_dir = os.path.join(format_dir(root_dir, game_config), 'eval')

    train_summary_writer, eval_summary_writer = get_writers_train_eval(summary_dir, eval_dir, game_config)
    eval_metrics = get_metrics_eval(num_parallel_environments, num_eval_episodes)
    eval_summary_writer_flush_op = eval_summary_writer.flush()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf.compat.v1.set_random_seed(random_seed)



        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    # ################################################ #
    # ---------------- Create Networks --------------- #
    # ################################################ #
        if use_rnns:
            actor_net, value_net = get_rnn_networks(tf_env, None)
        else:
            actor_net, value_net = get_networks(tf_env, {"actor_net": actor_fc_layers, "value_net": value_fc_layers})

    # ################################################ #
    # ---------------- Create PPO Agent -------------- #
    # ################################################ #
        tf_agent = ppo_agent.PPOAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            optimizer,
            entropy_regularization=0.2,
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step,
            normalize_observations=False)  # cause the observations also include the 0-1 mask

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)

        eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

    # ################################################ #
    # ---------------- Create Metrics ---------------- #
    # ################################################ #
        train_metrics, step_metrics, environment_steps_count = get_metrics_train_and_step(num_eval_episodes, num_parallel_environments)

        # Add to replay buffer and other agent specific observers.
        replay_buffer_observer = [replay_buffer.add_batch]

    # ################################################ #
    # ----------------- Trajectories ----------------- #
    # ################################################ #
        collect_policy = tf_agent.collect_policy
        collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=replay_buffer_observer + train_metrics,
            num_episodes=collect_episodes_per_iteration).run()

        trajectories = replay_buffer.gather_all()
        train_op, _ = tf_agent.train(experience=trajectories)

        with tf.control_dependencies([train_op]):
            clear_replay_op = replay_buffer.clear()

        with tf.control_dependencies([clear_replay_op]):
            train_op = tf.identity(train_op)

    # ################################################ #
    # ------------ Create Checkpointers -------------- #
    # ################################################ #
        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=tf_agent.policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)

        # ################################################ #
        # -------------- Create Summary Ops -------------- #
        # ################################################ #
        summary_ops = []
        for train_metric in train_metrics:
            summary_ops.append(train_metric.tf_summaries(
                train_step=global_step, step_metrics=step_metrics))

        with eval_summary_writer.as_default(), \
             tf.compat.v2.summary.record_if(True):
            for eval_metric in eval_metrics:
                eval_metric.tf_summaries(
                    train_step=global_step, step_metrics=step_metrics)

        init_agent_op = tf_agent.initialize()

        # ################################################ #
        # --------------- Initialize Graph --------------- #
        # ################################################ #

        with tf.compat.v1.Session(tf_master) as sess:
            train_checkpointer.initialize_or_restore(sess)
            rb_checkpointer.initialize_or_restore(sess)
            common.initialize_uninitialized_variables(sess)

            sess.run(init_agent_op)
            sess.run(train_summary_writer.init())
            sess.run(eval_summary_writer.init())

            collect_time = 0
            train_time = 0
            timed_at_step = sess.run(global_step)
            steps_per_second_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(), name='steps_per_sec_ph')
            steps_per_second_summary = tf.compat.v2.summary.scalar(
                name='global_steps_per_sec', data=steps_per_second_ph,
                step=global_step)

            # ################################################ #
            # -------------------- Loop ------ --------------- #
            # ------------ Collect/Train/Write --------------- #
            # ################################################ #

            while sess.run(environment_steps_count) < num_environment_steps:
                global_step_val = sess.run(global_step)
                if global_step_val % eval_interval == 0:
                    metric_utils.compute_summaries(
                        eval_metrics,
                        eval_py_env,
                        eval_py_policy,
                        num_episodes=num_eval_episodes,
                        global_step=global_step_val,
                        callback=eval_metrics_callback,
                        log=True,
                    )
                    sess.run(eval_summary_writer_flush_op)

                start_time = time.time()
                sess.run(collect_op)
                collect_time += time.time() - start_time
                start_time = time.time()
                total_loss, _ = sess.run([train_op, summary_ops])
                train_time += time.time() - start_time

                # ################################################ #
                # ---------- Logging and Checkpointing ----------- #
                # ################################################ #
                global_step_val = sess.run(global_step)
                if global_step_val % log_interval == 0:
                    logging.info('step = %d, loss = %f', global_step_val, total_loss)
                    steps_per_sec = (
                            (global_step_val - timed_at_step) / (collect_time + train_time))
                    logging.info('%.3f steps/sec', steps_per_sec)
                    sess.run(
                        steps_per_second_summary,
                        feed_dict={steps_per_second_ph: steps_per_sec})
                    logging.info('%s', 'collect_time = {}, train_time = {}'.format(
                        collect_time, train_time))
                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)

                if global_step_val % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step_val)

            # One final eval before exiting.
            metric_utils.compute_summaries(
                eval_metrics,
                eval_py_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=global_step_val,
                callback=eval_metrics_callback,
                log=True,
            )
            sess.run(eval_summary_writer_flush_op)
        tf.reset_default_graph()


def main(_):
    tf.compat.v1.enable_resource_variables()
    if tf.executing_eagerly():
        # self.skipTest('b/123777119')  # Secondary bug: ('b/123775375')
        return
    # loop over game params to create different configs
    logging.set_verbosity(logging.INFO)
    # todo: when this training is done, try different learning rates and architectures
    for colors in COLORS:
        for ranks in RANKS:
            for num_players in NUM_PLAYERS:
                for hand_size in HAND_SIZES:
                    for max_information_tokens in MAX_INFORMATION_TOKENS:
                        for max_life_tokens in MAX_LIFE_TOKENS:  # 2 * 1 * 1 * 4 * 4 * 2 = 64 total iterations
                            for custom_reward in CUSTOM_REWARDS:
                                for penalty in PENALTIES_LAST_HINT_TOKEN:
                                    config = {
                                        "colors": colors,
                                        "ranks": ranks,
                                        "players": num_players,
                                        "hand_size": hand_size,
                                        "max_information_tokens": max_information_tokens,
                                        "max_life_tokens": max_life_tokens,
                                        "observation_type": OBSERVATION_TYPE,
                                        "custom_reward": custom_reward,
                                        "penalty_last_hint_token": penalty
                                    }
                                    # ################################################ #
                                    # --------------- Load Environments -------------- #
                                    # ################################################ #
                                    eval_py_env = parallel_py_environment.ParallelPyEnvironment(
                                        [lambda: load_hanabi_env(config)] * FLAGS.num_parallel_environments)

                                    tf_env = tf_py_environment.TFPyEnvironment(
                                        parallel_py_environment.ParallelPyEnvironment(
                                            [lambda: load_hanabi_env(config)] * FLAGS.num_parallel_environments))
                                    train_eval(
                                        root_dir=FLAGS.root_dir,
                                        summary_dir=FLAGS.summary_dir,
                                        game_config=config,
                                        tf_master=FLAGS.master,
                                        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
                                        env_load_fn=load_hanabi_env,
                                        num_environment_steps=FLAGS.num_environment_steps,
                                        num_parallel_environments=FLAGS.num_parallel_environments,
                                        num_epochs=FLAGS.num_epochs,
                                        collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
                                        num_eval_episodes=FLAGS.num_eval_episodes,
                                        use_rnns=FLAGS.use_rnns,
                                    eval_py_env=eval_py_env,
                                    tf_env=tf_env)
                                    del eval_py_env
                                    del tf_env


if __name__ == '__main__':
    #flags.mark_flag_as_required('root_dir')
    # Summaries will be written extra, so you can put them directly to dropbox without saving checkpoints there
    #flags.mark_flag_as_required('summary_dir')
    app.run(main)
