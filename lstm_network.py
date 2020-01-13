import numpy as np
import tensorflow as tf
from lstm_utils import *


def build_network(observation, masks, nactions, input_states, input_states_v=None,
                  input_fc_layers=[128], lstm_layers=[128, ],
                  nenvs=8, nsteps=128, v_net='shared',
                  layer_norm=False, noisy_fc=False, noisy_lstm=False, noisy_heads=False, ):
    print('Building network:')
    print('FC:', input_fc_layers, 'LSTM:', lstm_layers)
    print('noisy_fc : %s, noisy_lstm : %s, noisy_heads: %s' % (noisy_fc, noisy_lstm, noisy_heads))
    print('Value network is %s' % v_net)
    print('nenvs : %d, nsteps : %d' % (nenvs, nsteps))
    print('Layer norm :', layer_norm)
    noise_list = []
    h = observation
    if v_net == 'shared':
        if noisy_fc:
            h, noise = multilayer_fc_noisy(h, input_fc_layers, scope='fc_net', layer_norm=layer_norm)
        else:
            h = multilayer_fc(h, input_fc_layers, scope='fc_net', layer_norm=layer_norm)
            noise = []
        noise_list.extend(noise)
        if len(lstm_layers):
            if noisy_lstm:
                lstm_outputs, states, init_state, noise = multilayer_lstm_noisy(h, masks, input_states, lstm_layers,
                                                                                'lstm_net', nenvs, nsteps, layer_norm)

            else:
                lstm_outputs, states, init_state = multilayer_lstm(h, masks, input_states, lstm_layers,
                                                                   'lstm_net', nenvs, nsteps, layer_norm)
                noise = []
            noise_list.extend(noise)
            h = lstm_outputs
        else:
            states, init_state = None, None
        if noisy_heads:
            policy, noise_p = multilayer_fc_noisy(h, [nactions], scope='policy', activation=None)
            value, noise_v = multilayer_fc_noisy(h, [1], scope='value', activation=None, noise=noise_p)
            noise_list.extend(noise_p)

        else:
            policy = multilayer_fc(h, [nactions], scope='policy', activation=None)
            value = multilayer_fc(h, [1], scope='value', activation=None)

        return policy, value, noise_list, states, init_state, None, None

    elif v_net == 'copy':

        if noisy_fc:
            h_p, noise_p = multilayer_fc_noisy(h, input_fc_layers, scope='fc_net_p')
            h_v, noise_v = multilayer_fc_noisy(h, input_fc_layers, scope='fc_net_v', noise=noise_p)
        else:
            h_p = multilayer_fc(h, input_fc_layers, scope='fc_net_p')
            h_v = multilayer_fc(h, input_fc_layers, scope='fc_net_v')
            noise_p = noise_v = []
        noise_list.extend(noise_p)
        if len(lstm_layers):
            assert input_states_v is not None, 'Specify input states for value network'

            if noisy_lstm:
                lstm_outputs_p, states_p, init_state_p, noise_p = multilayer_lstm_noisy(h_p, masks,
                                                                                        input_states, lstm_layers,
                                                                                        'lstm_net_p', nenvs, nsteps,
                                                                                        layer_norm)
                lstm_outputs_v, states_v, init_state_v, noise_v = multilayer_lstm_noisy(h_v, masks,
                                                                                        input_states_v, lstm_layers,
                                                                                        'lstm_net_v', nenvs, nsteps,
                                                                                        layer_norm,
                                                                                        noise=noise_p)

            else:
                lstm_outputs_p, states_p, init_state_p = multilayer_lstm(h_p, masks, input_states, lstm_layers,
                                                                         'lstm_net_p', nenvs, nsteps, layer_norm)
                lstm_outputs_v, states_v, init_state_v = multilayer_lstm(h_v, masks, input_states_v, lstm_layers,
                                                                         'lstm_net_v', nenvs, nsteps, layer_norm)
                noise_v = noise_v = []

            noise_list.extend(noise_p)
            h_p = lstm_outputs_p
            h_v = lstm_outputs_v
        else:
            states_p, init_state_p = None, None
            states_v, init_state_v = None, None

        if noisy_heads:
            policy, noise_p = multilayer_fc_noisy(h_p, [nactions], scope='policy', activation=None)
            value, noise_v = multilayer_fc_noisy(h_v, [1], scope='value', activation=None, noise=noise_p)
            noise_list.extend(noise_p)
        else:
            policy = multilayer_fc(h_p, [nactions], scope='policy', activation=None, layer_norm=False)
            value = multilayer_fc(h_v, [1], activation=None, scope='value', layer_norm=False)

        return policy, value, noise_list, states_p, init_state_p, states_v, init_state_v


class Network:
    def __init__(self, obs, masks, legal_moves, nactions, nenvs, nsteps,
                 input_fc_layers=[128], lstm_layers=[128], v_net='shared',
                 layer_norm=False, noisy_fc=False, noisy_lstm=False, ):

        self.nactions = nactions
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.nbatch = (self.nsteps * self.nenvs)
        self.OBS = obs
        self.MASKS = masks
        self.LEGAL_MOVES = legal_moves

        if len(lstm_layers):
            self.input_states = [tf.placeholder(tf.float32, [self.nenvs, 2 * nlstm])
                                 for nlstm in lstm_layers]
            if v_net == 'shared':
                self.input_states_v = None
            elif v_net == 'copy':
                self.input_states_v = [tf.placeholder(tf.float32, [self.nenvs, 2 * nlstm])
                                       for nlstm in lstm_layers]
        else:
            self.input_states = None
            self.input_states_v = None
        policy, value, noise_list, states, init_state, states_v, init_state_v = build_network(self.OBS, self.MASKS,
                                                                                              nactions,
                                                                                              self.input_states,
                                                                                              self.input_states_v,
                                                                                              input_fc_layers,
                                                                                              lstm_layers,
                                                                                              nenvs, nsteps,
                                                                                              v_net, layer_norm,
                                                                                              noisy_fc,
                                                                                              noisy_lstm,
                                                                                              )
        self.states = states
        self.init_state = init_state
        self.states_v = states_v
        self.init_state_v = init_state_v
        if self.states is None:
            self.states = tf.constant([])
            self.states_v = tf.constant([])

        self.policy = policy + legal_moves
        self.value = value
        self.noise_list = noise_list

        self.probs = tf.nn.softmax(self.policy)
        self.logp = tf.nn.log_softmax(self.policy)

        self.sample_action = tf.squeeze(tf.multinomial(self.policy, 1), axis=1)
        self.sample_action_oh = tf.one_hot(self.sample_action, depth=nactions)

        self.alogp = tf.reduce_sum(self.sample_action_oh * self.logp, axis=1)

    def step(self, sess, obs, legal_moves, masks, inp_states=None, inp_states_v=None, noise=None):
        # feed values
        feed_dict = {self.OBS: obs, self.LEGAL_MOVES: legal_moves, self.MASKS: masks}
        for noise_ph, noise_val in zip(self.noise_list, noise):
            feed_dict[noise_ph] = noise_val
        if inp_states is not None:
            for S, s_val in zip(self.input_states, inp_states):
                feed_dict[S] = s_val
            if inp_states_v is not None:
                for S, s_val in zip(self.input_states_v, inp_states_v):
                    feed_dict[S] = s_val
            feed_dict[self.MASKS] = masks
        if self.states_v is not None:
            actions, probs, alogps, values, states, states_v = sess.run([self.sample_action, self.probs,
                                                                         self.alogp, self.value,
                                                                         self.states, self.states_v],
                                                                        feed_dict)
        else:
            actions, probs, alogps, values, states = sess.run([self.sample_action, self.probs,
                                                               self.alogp, self.value,
                                                               self.states, ], feed_dict)
            states_v = None
        return actions, probs, alogps, values, states, states_v


