import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents.networks

from tensorflow.python.keras.engine.network import Network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.ddpg import actor_rnn_network

import numpy as np


class MaskedActorDistributionNetwork(ActorDistributionNetwork):
    """An actor network which filters the output action distribution using a mask.

    The mask is an np.ndarray which is -np.inf if the action is not possible
    and 0 if it is possible. By adding this mask to the logits (the vector before the softmax),
    the invalid actions are now -inf and only valid actions remain with the same value.

    The mask is stored inside the observation in order to have an association between the two.
    The observation is therefore of the form { 'state': actual_observation, 'mask': mask }
    """

    def __init__(self, input_tensor_spec, output_tensor_spec, fc_layer_params):
        super().__init__(input_tensor_spec['state'], output_tensor_spec, fc_layer_params)

    def call(self, observations, step_type, network_state):
        states = observations['state']
        masks = observations['mask']

        # print('MASKS', masks)
        # print('STATES', states)

        action_distributions, new_network_states = super().call(
            states, step_type, network_state)

        # for some reason, when we get a batch there is an extra axis of dimension 1
        # we therefore also have to insert this axis into our mask
        if len(action_distributions.logits.shape) == 4:
            masks = tf.expand_dims(masks, 2)

        # set logits to -inf if their corresponding move is illegal
        # masks = masks.astype(np.float32)
        # neginfs = tf.convert_to_tensor(np.full(action_distributions.logits.shape, np.NINF), dtype='float')
        # masked_logits = tf.where(condition=masks, x=action_distributions.logits, y=neginfs)
        masked_logits = tf.add(action_distributions.logits, masks)

        # the dtype doesn't refer to the logits
        # but the action that is then created from the distribution
        return tfp.distributions.Categorical(
            logits=masked_logits, dtype=tf.int64), new_network_states

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)


class MaskedValueNetwork(ValueNetwork):
    """A value network which uses only observation['state'] as observation.

    For actor-critic methods, the value network gets the same input
    as the actor network however only the actor network actually
    needs the mask, so in the value network we have to throw it away explicitly
    """

    def __init__(self, input_tensor_spec, fc_layer_params):
        super().__init__(input_tensor_spec['state'], fc_layer_params=fc_layer_params)

    def call(self, observation, step_type=None, network_state=()):
        return super().call(observation['state'], step_type, network_state)

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)


class MaskedActorDistributionRnnNetwork(ActorDistributionRnnNetwork):

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 input_fc_layer_params=(200, 100),
                 output_fc_layer_params=None,
                 lstm_size=(40,), ):
        super(MaskedActorDistributionRnnNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec['state'],
            output_tensor_spec=output_tensor_spec,
            input_fc_layer_params=input_fc_layer_params,
            output_fc_layer_params=output_fc_layer_params,
            lstm_size=lstm_size)

    def call(self, observations, step_type, network_state):
        states = observations['state']
        masks = observations['mask']

        # print('MASKS', masks)
        # print('STATES', states)

        action_distributions, new_network_states = super().call(
            states, step_type, network_state)

        # for some reason, when we get a batch there is an extra axis of dimension 1
        # we therefore also have to insert this axis into our mask
        if len(action_distributions.logits.shape) == 4:
            masks = tf.expand_dims(masks, 2)

        # set logits to -inf if their corresponding move is illegal
        # masks = masks.astype(np.float32)
        # neginfs = tf.convert_to_tensor(np.full(action_distributions.logits.shape, np.NINF), dtype='float')
        # masked_logits = tf.where(condition=masks, x=action_distributions.logits, y=neginfs)
        masked_logits = tf.add(action_distributions.logits, masks)

        # the dtype doesn't refer to the logits
        # but the action that is then created from the distribution
        return tfp.distributions.Categorical(
            logits=masked_logits, dtype=tf.int64), new_network_states

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)


class MaskedValueRnnNetwork(ValueRnnNetwork):
    """A value network which uses only observation['state'] as observation.

    For actor-critic methods, the value network gets the same input
    as the actor network however only the actor network actually
    needs the mask, so in the value network we have to throw it away explicitly
    """

    def __init__(self, input_tensor_spec, input_fc_layer_params, output_fc_layer_params=None, lstm_size=(40,)):
        super().__init__(input_tensor_spec['state'],
                         input_fc_layer_params=input_fc_layer_params,
                         output_fc_layer_params=output_fc_layer_params,
                         lstm_size=lstm_size)

    def call(self, observation, step_type=None, network_state=()):
        return super().call(observation['state'], step_type, network_state)

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)


class MaskedCriticRnnNetwork(critic_rnn_network.CriticRnnNetwork):
    """A critic network which uses only observation['state'] as observation, to use with ddpg agents.

    For actor-critic methods, the critic network gets the same input
    as the actor network however only the actor network actually
    needs the mask, so in the value network we have to throw it away explicitly
    """

    def __init__(self,
                 input_tensor_spec,
                 observation_fc_layer_params,
                 action_fc_layer_params,
                 joint_fc_layer_params,
                 lstm_size,
                 output_fc_layer_params=None
                 ):
        super().__init__(input_tensor_spec['state'],
                         observation_fc_layer_params=observation_fc_layer_params,
                         action_fc_layer_params=action_fc_layer_params,
                         joint_fc_layer_params=joint_fc_layer_params,
                         lstm_size=lstm_size,
                         output_fc_layer_params=output_fc_layer_params)

    def call(self, observation, step_type=None, network_state=()):
        return super().call(observation['state'], step_type, network_state)

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)


class MaskedActorRnnNetwork(actor_rnn_network.ActorRnnNetwork):

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 conv_layer_params=None,
                 input_fc_layer_params=(200, 100),
                 lstm_size=(40,),
                 output_fc_layer_params=(200, 100),
                 activation_fn=tf.keras.activations.relu,
                 name='ActorRnnNetwork'):
        super(MaskedActorRnnNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec['state'],
            output_tensor_spec=output_tensor_spec,
            conv_layer_params=conv_layer_params,
            input_fc_layer_params=input_fc_layer_params,
            output_fc_layer_params=output_fc_layer_params,
            lstm_size=lstm_size,
            activation_fn=activation_fn,
            name=name)

    def call(self, observation, step_type, network_state):
        states = observation['state']
        mask = observation['mask']

        # print('MASKS', masks)
        # print('STATES', states)

        actions, new_network_states = super().call(
            states, step_type, network_state)

        # for some reason, when we get a batch there is an extra axis of dimension 1
        # we therefore also have to insert this axis into our mask
        if len(actions.shape) == 4:
            mask = tf.expand_dims(mask, 2)

        masked_actions = tf.add(actions, mask)

        # the dtype doesn't refer to the logits
        # but the action that is then created from the distribution
        return masked_actions, new_network_states

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)
