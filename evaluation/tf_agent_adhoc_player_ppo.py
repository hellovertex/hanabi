from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_policy
from tf_agents.networks.value_network import ValueNetwork
from tensorflow.python.keras.engine.network import Network

# own imports
import training.tf_agents_lib.masked_networks as masked_networks
from training.tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper
import hanabi_learning_environment.rl_env as rl_env

class PPOTfAgentAdHocPlayer(object):

  def __init__(self,
      root_dir,
      game_type,
      num_players,
      actor_fc_layers=(150, 75), 
      value_fc_layers=(150, 75), 
      actor_fc_layers_rnn=(150,),
      value_fc_layers_rnn=(150,),
      lstm_size=(75,75),
      use_rnns=False,):

    tf.reset_default_graph()
    self.sess = tf.Session()
    tf.compat.v1.enable_resource_variables()
    
    pyhanabi_env = rl_env.make(environment_name=game_type, num_players=num_players)
    py_env = PyhanabiEnvWrapper(pyhanabi_env)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    with self.sess.as_default():
      # init the agent
      if use_rnns:
        actor_net = masked_networks.MaskedActorDistributionRnnNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            input_fc_layer_params=actor_fc_layers_rnn,
            output_fc_layer_params=None, 
            lstm_size=lstm_size,)
        value_net = MaskedValueNetwork(
            tf_env.observation_spec(), fc_layer_params=value_fc_layers)
      else:
        actor_net = masked_networks.MaskedActorDistributionNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=actor_fc_layers)
        value_net = MaskedValueNetwork(
            tf_env.observation_spec(), fc_layer_params=value_fc_layers)

      global_step = tf.compat.v1.train.get_or_create_global_step() # necessary ???
      tf_agent = ppo_agent.PPOAgent(
          tf_env.time_step_spec(),
          tf_env.action_spec(),
          actor_net=actor_net,
          value_net=value_net,
          train_step_counter=global_step,
          normalize_observations=False) # cause the observations also include the 0-1 mask

      
      self.policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

      # load checkpoint
      train_dir = os.path.join(root_dir, 'train')
      self.policy.initialize(None)
      self.policy.restore(root_dir)
      init_agent_op = tf_agent.initialize()

      self.sess.run(init_agent_op)


  def act(self, observation):
    with self.sess.as_default():
      #print(observation)
      policy_step = self.policy.action(observation)
      return policy_step.action


class MaskedValueNetwork(ValueNetwork):
    """A value network which uses only observation['state'] as observation.

    For actor-critic methods, the value network gets the same input
    as the actor network however only the actor network actually
    needs the mask, so in the value network we have to throw it away explicitly
    """

    def __init__(self, input_tensor_spec, fc_layer_params):
        super().__init__(input_tensor_spec['state'], fc_layer_params=fc_layer_params)

    def call(self, observation, step_type=None, network_state=()):
        return super().call(observation, step_type, network_state)

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)