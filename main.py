from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# project imports
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.replay_buffers import replay_buffer
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from pyhanabi_env_wrapper import PyhanabiEnvWrapper
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent, element_wise_squared_loss
import tensorflow as tf
import test
from policy_wrapper import LegalMovesSampler
import utils
import rl_env
tf.compat.v1.enable_v2_behavior()

""" TRAIN HYPERPARAMS """
env_name = 'CartPole-v0'  # @param
num_iterations = 20000  # @param

initial_collect_steps = 1000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 100000  # @param

fc_layer_params = (100,)

batch_size = 64  # @param
learning_rate = 1e-3  # @param
log_interval = 200  # @param

num_eval_episodes = 10  # @param
eval_interval = 1000  # @param

""" ENVIRONMENT """
# game config
variant = "Hanabi-Full"
num_players = 5

# load and wrap environment, to use it with TF-Agent library
pyhanabi_env = rl_env.make(environment_name=variant, num_players=num_players)
env = PyhanabiEnvWrapper(pyhanabi_env)
# test.validate_py_environment(env)

""" DQN AGENT """
# init feedforward net
q_net = QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=fc_layer_params)
flat_action_spec = tf.nest.flatten(env.action_spec())
# init optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)

# init dqn agent
#print("ACTION SPEC", flat_action_spec)
#print(flat_action_spec[0])
#print(flat_action_spec[0].shape)
#print(flat_action_spec[0].shape.ndims)
"""tf_agent = DqnAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=element_wise_squared_loss,
    train_step_counter=train_step_counter
)
tf_agent.initialize()

# init q policy
eval_policy = LegalMovesSampler(tf_agent.policy, env)

# run simple test
utils.compute_avg_return(env, eval_policy, num_eval_episodes)
"""

actor_net = actor_distribution_network.ActorDistributionNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params
)

tf_agent = reinforce_agent.ReinforceAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter
)
eval_policy = LegalMovesSampler(tf_agent.policy, env)


def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1

# Optional wrapping some code in a graph using TF functoin
tf_agent.train = common.function(tf_agent.train)

# reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agents policy once before training
avg_return = utils.compute_avg_return(env, eval_policy)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      env, eval_policy, 2)

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  train_loss = tf_agent.train(experience)
  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = utils.compute_avg_return(env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)