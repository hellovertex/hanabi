from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_policy
from tf_agents.networks.value_network import ValueNetwork
from tensorflow.python.keras.engine.network import Network
from tf_agents.agents.reinforce import reinforce_agent

# own imports
from training.tf_agents_lib import masked_networks
from training.tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper
import hanabi_learning_environment.rl_env as rl_env


class ReinforceTfAgentAdHocPlayer(object):

    def __init__(self,
                 root_dir,
                 game_type='Hanabi-Full',
                 num_players=4,
                 actor_fc_layers=(100, ),
                 value_fc_layers=(100, ),
                 use_value_network=False
                 ):
        tf.reset_default_graph()
        self.sess = tf.Session()
        tf.compat.v1.enable_resource_variables()

        pyhanabi_env = rl_env.make(environment_name=game_type, num_players=num_players)
        py_env = PyhanabiEnvWrapper(pyhanabi_env)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        with self.sess.as_default():
            # init the agent
            actor_net = masked_networks.MaskedActorDistributionNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                fc_layer_params=actor_fc_layers
            )
            value_network = None
            if use_value_network:
                value_network = MaskedValueNetwork(
                    tf_env.observation_spec(),
                    fc_layer_params=value_fc_layers
                )

            global_step = tf.compat.v1.train.get_or_create_global_step()  # necessary ??? => Yes baby

            tf_agent = reinforce_agent.ReinforceAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                actor_network=actor_net,
                value_network=value_network if use_value_network else None,
                value_estimation_loss_coef=.2,
                gamma=.9,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
                debug_summaries=False,
                summarize_grads_and_vars=False,
                train_step_counter=global_step
                )

            self.policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

            # load checkpoint
            #train_dir = os.path.join(root_dir, 'train')
            self.policy.initialize(None)
            self.policy.restore(root_dir)
            init_agent_op = tf_agent.initialize()

            self.sess.run(init_agent_op)

    def act(self, observation):
        with self.sess.as_default():
            # print(observation)
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