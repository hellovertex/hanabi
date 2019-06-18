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
import policy_wrapper
tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.ddpg import critic_network
import sac_agent_custom
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import epsilon_greedy_policy
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

# initial_collect_steps = 10000  # @param
initial_collect_steps = 100  # @param
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
import test
# game config
variant = "Hanabi-Full"
num_players = 5

# load and wrap environment, to use it with TF-Agent library
pyhanabi_env_train = rl_env.make(environment_name=variant, num_players=num_players)
pyhanabi_env_eval = rl_env.make(environment_name=variant, num_players=num_players)
py_env_train = PyhanabiEnvWrapper(pyhanabi_env_train)
py_env_eval = PyhanabiEnvWrapper(pyhanabi_env_eval)
# check specs after wrapping env
test.validate_py_environment(py_env_train)