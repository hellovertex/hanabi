""" IMPORTS """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image

import tensorflow as tf

tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# ------------------------------------------------------------------------------- #
""" HYPERPARAMS """
# ------------------------------------------------------------------------------- #

env_name = 'MinitaurBulletEnv-v0'  # @param
num_iterations = 1000000  # @param

initial_collect_steps = 10000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 1000000  # @param

batch_size = 256  # @param

critic_learning_rate = 3e-4  # @param
actor_learning_rate = 3e-4  # @param
alpha_learning_rate = 3e-4  # @param
target_update_tau = 0.005  # @param
target_update_period = 1  # @param
gamma = 0.99  # @param
reward_scale_factor = 1.0  # @param
gradient_clipping = None  # @param

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 10  # @param

num_eval_episodes = 30  # @param
eval_interval = 10000  # @param

# ------------------------------------------------------------------------------- #
""" ENVIRONMENT """
# ------------------------------------------------------------------------------- #

import rl_env
import utils
from pyhanabi_env_wrapper import PyhanabiEnvWrapper

# game config
variant = "Hanabi-Full"
num_players = 5

# load and wrap environment, to use it with TF-Agent library
pyhanabi_env = rl_env.make(environment_name=variant, num_players=num_players)
py_env = PyhanabiEnvWrapper(pyhanabi_env)
# check specs after wrapping env
# test.validate_py_environment(py_env)
train_env = tf_py_environment.TFPyEnvironment(py_env)
# TFPyEnvironment -> BatchedPyEnvironment -> PyEnvironmentBaseWrapper -> pyhanabi_env
py_env = train_env._env.envs[0].wrapped_env()
eval_env = train_env
observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()

# ------------------------------------------------------------------------------- #
""" NETWORKS (ACTOR AND CRITIC)"""
# ------------------------------------------------------------------------------- #

critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params)


def normal_projection_net(action_spec, init_means_output_factor=0.1):
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


import actor_distribution_network_custom

# actor_net = actor_distribution_network.ActorDistributionNetwork(
actor_net = actor_distribution_network_custom.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layer_params,
    continuous_projection_net=normal_projection_net,
    environment=py_env
)
# ------------------------------------------------------------------------------- #
""" SAC-AGENT INIT """
# ------------------------------------------------------------------------------- #

# the policy remains the same, as the agent calls it with its actor network
# it is therefore sufficient to pass the environment to a customized network
# with which the agent is then called
global_step = tf.compat.v1.train.get_or_create_global_step()

tf_agent = sac_agent.SacAgent(
    train_env.time_step_spec(),
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate),
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)
tf_agent.initialize()

# ------------------------------------------------------------------------------- #
""" POLICIES """
# ------------------------------------------------------------------------------- #

eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
collect_policy = tf_agent.collect_policy


# utils.compute_avg_return(train_env, eval_policy, num_eval_episodes)

def compute_avg_return(environment, policy, num_episodes=5):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# Please also see the metrics module for standard implementations of different
# metrics.
# ------------------------------------------------------------------------------- #
""" REPLAY BUFFER """
# ------------------------------------------------------------------------------- #
# for most agents, the collect_data_spec is a trajectory named tuple containing the
# observation, action, reward, etc.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

# ------------------------------------------------------------------------------- #
""" DATA COLLECTION """
# ------------------------------------------------------------------------------- #
# collects n steps or episodes on an environment using a specific policy
initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    env=train_env,
    policy=collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=initial_collect_steps
)
initial_collect_driver.run()

# In order to save space, we only store the current observation in each row of the
# replay buffer. But since the SAC Agent needs both the current and next observation
# to cocmpute the loss, we always sample two adjacent rows for each item in the bath
# by setting num_steps = 2
# Dataset generates trajectories with shapce [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2
).prefetch(3)

iterator = iter(dataset)

# ------------------------------------------------------------------------------- #
""" TRAINING the SAC-AGENT """
# ------------------------------------------------------------------------------- #

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)
collect_driver.run = common.function(collect_driver.run)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    print("FOO")

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_driver.run()
    print("BAR")

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
