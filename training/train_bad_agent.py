import os
import time

import tensorflow as tf

from absl import logging
from absl import app
from absl import flags

from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from custom_environment.pub_mdp import PubMDP
from custom_environment.pubmdp_env_wrapper import PubMDPWrapper
from hanabi_learning_environment import rl_env
from training.hanabi_small.train_ppo_custom_env import get_metrics_eval, get_writers_train_eval
from training.tf_agents_lib import pyhanabi_env_wrapper
from training.tf_agents_lib.masked_networks import MaskedActorDistributionNetwork, MaskedValueNetwork, \
    MaskedValueEvalNetwork
from training.tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper

FLAGS = flags.FLAGS

DEFAULT_CONFIG = {  # config for Hanabi-Small
        "colors": 2,
        "ranks": 5,
        "players": 2,
        "hand_size": 2,
        "max_information_tokens": 3,
        "max_life_tokens": 1,
        "observation_type": 1}

# use usepubmdp to generate one fix input
# input to network
# train using policy gradient loss
# try applying variable scope to access the network from within a python class and the graph and see if the weights are
# shared properly while updating them

# need to create weights and biases
PARAMS = {
    'game_config': DEFAULT_CONFIG,
    'num_parallel_environments': 30,
    'learning_rate': 1e-4,
    'num_epochs': 25,
    'debug_summaries': False,
    'summarize_grads_and_vars': False
}


def load_hle(game_config):
    assert isinstance(game_config, dict)

    pyhanabi_env = rl_env.HanabiEnv(game_config)

    if pyhanabi_env is not None:
        return pyhanabi_env_wrapper.PyhanabiEnvWrapper(pyhanabi_env)

    return None


def load_hanabi_pub_mdp(game_config, public_policy=None):
    assert isinstance(game_config, dict)
    env = PubMDP(game_config, public_policy)
    if env is not None:
        return PubMDPWrapper(env)
    return None


def get_obs_spec_action_spec_from_game_config(game_config):
    """
    Returns observation_spec and action_spec required by tf_agents.network
     Creates a pub mdp just to read out the observation spec and action spec for given game-config.
     These will be used to create the actor_net, as its input_spec and output_spec.
     The environment created here will not be used.

     Args:
         game_config: Used to create HanabiEnv object
     Returns:
         time_step_spec(), observation_spec(), action_spec()
     for wrapped tf_environment
     """
    gc_tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: load_hanabi_pub_mdp(game_config, public_policy=None)] * 1)
    )  # we create parallel environment just to make sure it is 1:1 identical with the one really used

    return gc_tf_env.time_step_spec(), gc_tf_env.observation_spec(), gc_tf_env.action_spec()


def get_metrics_train_and_step():
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    environment_steps_count = environment_steps_metric.result()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    return train_metrics, step_metrics, environment_steps_count


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

    return formatted_summary_dir


def train_eval(
        root_dir,
        summary_dir,
        game_config,
        tf_master='',
        random_seed=0,
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
        log_interval=50,
        summary_interval=50,
        summaries_flush_secs=1,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):

    # ################################################ #
    # ------------ Create summary-writers ------------ #
    # ################################################ #
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(format_dir(root_dir, game_config), 'train')
    eval_dir = os.path.join(format_dir(root_dir, game_config), 'eval')
    if summary_dir is None:
        summary_dir = train_dir
    train_summary_writer, eval_summary_writer = get_writers_train_eval(summary_dir, eval_dir, game_config)
    eval_metrics = get_metrics_eval(num_parallel_environments, num_eval_episodes)
    eval_summary_writer_flush_op = eval_summary_writer.flush()

    # ################################################ #
    # ------------ global step, record_if ------------ #
    # ################################################ #
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf.compat.v1.set_random_seed(random_seed)

        # ################################################ #
        # ---------------- Create Networks --------------- #
        # ################################################ #
        time_step_spec, observation_spec, action_spec = get_obs_spec_action_spec_from_game_config(game_config)
        actor_net = MaskedActorDistributionNetwork(observation_spec, action_spec, fc_layer_params=(384, 384))
        value_net = MaskedValueNetwork(observation_spec, fc_layer_params=(384, 384))

        # ################################################ #
        # ---------------- Create PPO Agent -------------- #
        # ################################################ #
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        agent_BAD = PPOAgent(
                    time_step_spec,
                    action_spec,
                    optimizer,
                    actor_net=actor_net,
                    value_net=value_net,
                    num_epochs=num_epochs,
                    debug_summaries=debug_summaries,
                    summarize_grads_and_vars=summarize_grads_and_vars,
                    train_step_counter=global_step,
                    normalize_observations=False)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent_BAD.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)

        eval_py_policy = py_tf_policy.PyTFPolicy(agent_BAD.policy)

        # ################################################ #
        # --------------- Load Environments -------------- #
        # ################################################ #
        """ Note: We create environments using agent_BADs collect policy because this is in fact the public policy.
         The train_op is currently agent.train() but will be replaced using learner.train(actor_net)
         where actor_net is used by the collect (public) policies of the BAD agents
         """
        env_policy = py_tf_policy.PyTFPolicy(agent_BAD.collect_policy, seed=123)
        # todo this blows up the value network, maybe pass tf sess and exec eagerly
        tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: load_hanabi_pub_mdp(DEFAULT_CONFIG, public_policy=agent_BAD.collect_policy)]
                * PARAMS['num_parallel_environments'])
        )
        # this will be a normal HLE without a public agent, as
        eval_py_env = parallel_py_environment.ParallelPyEnvironment(
            # [lambda: load_hle(game_config)] * num_parallel_environments
            [lambda: load_hanabi_pub_mdp(DEFAULT_CONFIG, public_policy=env_policy)]
            * PARAMS['num_parallel_environments']
        )

        # ################################################ #
        # ---------------- Create Metrics ---------------- #
        # ################################################ #
        train_metrics, step_metrics, environment_steps_count = get_metrics_train_and_step()

        # Add to replay buffer and other agent specific observers.
        replay_buffer_observer = [replay_buffer.add_batch]

        # ################################################ #
        # ----------------- Trajectories ----------------- #
        # ################################################ #
        collect_policy = agent_BAD.collect_policy
        collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=replay_buffer_observer + train_metrics,
            num_episodes=collect_episodes_per_iteration).run()

        trajectories = replay_buffer.gather_all()
        train_op, _ = agent_BAD.train(experience=trajectories)

        with tf.control_dependencies([train_op]):
            clear_replay_op = replay_buffer.clear()

        with tf.control_dependencies([clear_replay_op]):
            train_op = tf.identity(train_op)

        # ################################################ #
        # ------------ Create Checkpointers -------------- #
        # ################################################ #
        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=agent_BAD,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=agent_BAD.policy,
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

        init_agent_op = agent_BAD.initialize()
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
        return
    # loop over game params to create different configs
    logging.set_verbosity(logging.INFO)
    train_eval(
        root_dir=FLAGS.root_dir,
        summary_dir=FLAGS.summary_dir,
        game_config=DEFAULT_CONFIG,
        tf_master=FLAGS.master,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        num_environment_steps=FLAGS.num_environment_steps,
        num_parallel_environments=FLAGS.num_parallel_environments,
        num_epochs=FLAGS.num_epochs,
        collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
        num_eval_episodes=FLAGS.num_eval_episodes,
        )

if __name__ == '__main__':
    app.run(main)