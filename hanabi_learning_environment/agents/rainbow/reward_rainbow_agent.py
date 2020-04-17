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
"""Implementation of a Rainbow agent adapted to the multiplayer setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import rainbow_agent
import gin.tf
import numpy as np
import prioritized_replay_memory
import tensorflow as tf


slim = tf.contrib.slim

class Reward_RainbowAgent(rainbow_agent.RainbowAgent):
  """A compact implementation of the multiplayer Rainbow agent."""

  @gin.configurable
  def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               num_atoms=51,
               vmax=25.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=500,
               update_period=4,
               target_update_period=500,
               epsilon_train=0.0,
               epsilon_eval=0.0,
               epsilon_decay_period=1000,
               learning_rate=0.000025,
               optimizer_epsilon=0.00003125,
               tf_device='/cpu:*',
               tf_session=None):
    """Initializes the agent and constructs its graph.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_size: int, size of observation vector.
      num_players: int, number of players playing this game.
      num_atoms: Int, the number of buckets for the value function distribution.
      vmax: float, maximum return predicted by a value distribution.
      gamma: float, discount factor as commonly used in the RL literature.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of stored transitions before training.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_train: float, final epsilon for training.
      epsilon_eval: float, epsilon during evaluation.
      epsilon_decay_period: int, number of steps for epsilon to decay.
      learning_rate: float, learning rate for the optimizer.
      optimizer_epsilon: float, epsilon for Adam optimizer.
      tf_device: str, Tensorflow device on which to run computations.
    """
    #first construct RainbowAgent
    super(Reward_RainbowAgent, self).__init__(
        num_actions=num_actions,
        observation_size=observation_size,
        num_players=num_players,
        num_atoms=num_atoms,
        vmax=vmax.,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        learning_rate=learning_rate,
        optimizer_epsilon=optimizer_epsilon,
        tf_device=tf_device,
        tf_session=tf_session)
