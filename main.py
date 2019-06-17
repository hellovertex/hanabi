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

""" HYPERPARAMS """
env_name = 'MinitaurBulletEnv-v0'  # @param
num_iterations = 1000000 # @param

initial_collect_steps = 10000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 1000000 # @param

batch_size = 256  # @param

critic_learning_rate = 3e-4  # @param
actor_learning_rate = 3e-4  # @param
alpha_learning_rate = 3e-4 # @param
target_update_tau = 0.005 # @param
target_update_period = 1 #@param
gamma = 0.99 #@param
reward_scale_factor = 1.0 #@param
gradient_clipping = None #@param

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000  # @param

num_eval_episodes = 30  # @param
eval_interval = 10000  # @param

""" ENVIRONMENT """
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

observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params)


def normal_projection_net(action_spec,init_means_output_factor=0.1):
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


eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
collect_policy = tf_agent.collect_policy


utils.compute_avg_return(train_env, eval_policy, num_eval_episodes)
