#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""Implementation of a Rainbow agent adapted to the multiplayer setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import dqn_agent
import gin.tf
import numpy as np
import prioritized_replay_memory
import tensorflow as tf


slim = tf.contrib.slim


@gin.configurable
class RainbowAgentNoDist(dqn_agent.DQNAgent):
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
               tf_device='/cpu:*'):
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

    self.learning_rate = learning_rate
    self.optimizer_epsilon = optimizer_epsilon

    super(RainbowAgentNoDist, self).__init__(
        num_actions=num_actions,
        observation_size=observation_size,
        num_players=num_players,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device)
    tf.logging.info('\t learning_rate: %f', learning_rate)
    tf.logging.info('\t optimizer_epsilon: %f', optimizer_epsilon)

  def _build_replay_memory(self, use_staging):
    """Creates the replay memory used by the agent.

    Rainbow uses prioritized replay.

    Args:
      use_staging: bool, whether to use a staging area in the replay memory.

    Returns:
      A replay memory object.
    """
    return prioritized_replay_memory.WrappedPrioritizedReplayMemory(
        num_actions=self.num_actions,
        observation_size=self.observation_size,
        stack_size=1,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma)


  def _build_train_op(self):
    """Builds the training op for Rainbow.

    Returns:
      train_op: An op performing one step of training.
    """

    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_qs * replay_action_one_hot,
        reduction_indices=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)

    update_priorities_op = self._replay.tf_set_priority(
        self._replay.indices, tf.sqrt(loss + 1e-10))

    target_priorities = self._replay.tf_get_priority(self._replay.indices)
    target_priorities = tf.math.add(target_priorities, 1e-10)
    target_priorities = 1.0 / tf.sqrt(target_priorities)
    target_priorities /= tf.reduce_max(target_priorities)

    weighted_loss = target_priorities * loss

    with tf.control_dependencies([update_priorities_op]):
        return self.optimizer.minimize(tf.reduce_mean(weighted_loss)), weighted_loss

    # return self.optimizer.minimize(tf.reduce_mean(loss))
"""
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_logits, reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate,
        epsilon=self.optimizer_epsilon)

    update_priorities_op = self._replay.tf_set_priority(
        self._replay.indices, tf.sqrt(loss + 1e-10))

    target_priorities = self._replay.tf_get_priority(self._replay.indices)
    target_priorities = tf.math.add(target_priorities, 1e-10)
    target_priorities = 1.0 / tf.sqrt(target_priorities)
    target_priorities /= tf.reduce_max(target_priorities)

    weighted_loss = target_priorities * loss

    with tf.control_dependencies([update_priorities_op]):
      return optimizer.minimize(tf.reduce_mean(weighted_loss)), weighted_loss
"""
