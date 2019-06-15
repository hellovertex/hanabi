import rl_env
from pyhanabi_env_wrapper import PyhanabiEnvWrapper
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent, element_wise_squared_loss
import tensorflow as tf
import test
from policy_wrapper import LegalMovesSampler
import utils

""" TRAIN HYPERPARAMS """
fc_layer_params = (100,)
learning_rate = 1e-3
num_eval_episodes = 10

""" ENVIRONMENT """
# game config
variant = "Hanabi-Full"
num_players = 5

# load and wrap environment, to use it with TF-Agent library
pyhanabi_env = rl_env.make(environment_name=variant, num_players=num_players)
env = PyhanabiEnvWrapper(pyhanabi_env)
test.validate_py_environment(env)

""" DQN AGENT """
# init feedforward net
q_net = QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params
)
# init optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)

# init dqn agent
tf_agent = DqnAgent(
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

""" PPO AGENT """
