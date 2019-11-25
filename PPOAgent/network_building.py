import numpy as np
import tensorflow as tf
from scipy.special import softmax
from .util import *

def build_network(observation, num_actions, M, input_states, input_fc_layers = [256, 256], lstm_layers = [],
                  output_fc_layers = [], nenvs = 8, nminibatches = 1, nsteps = 128, v_net = 'shared', noisy_fc = False,
                  noisy_lstm = False, input_states_v = None, layer_norm = True):
    
    noise_list = []
    h = observation
    if v_net == 'shared':
        if noisy_fc:
            h, noise = multilayer_fc_noisy(h, input_fc_layers, scope = 'fc_net', layer_norm = layer_norm)
        else:
            h = multilayer_fc(h, input_fc_layers, scope = 'fc_net', layer_norm = layer_norm)
            noise = []
        noise_list.extend(noise)
        if len(lstm_layers):
            if noisy_lstm:
                lstm_outputs, states, init_state, noise = multilayer_lstm_noisy(h, M, input_states, lstm_layers, 
                                                                                'lstm_net', nenvs // nminibatches, nsteps)
                
            else:
                lstm_outputs, states, init_state = multilayer_lstm(h, M, input_states, lstm_layers, 
                                                                   'lstm_net', nenvs // nminibatches, nsteps)
                noise = []
            noise_list.extend(noise)
            h = lstm_outputs
        else:
            states, init_state = None, None
            
        if len(output_fc_layers):
            if noisy:
                h, noise = multilayer_fc_noisy(h, output_fc_layers, scope = 'fc_net', layer_norm = layer_norm)
            else:
                h = multilayer_fc(h, output_fc_layers, scope = 'fc_net', layer_norm = layer_norm)

        policy = multilayer_fc(h, [num_actions], scope = 'policy', activation = None)
        value = multilayer_fc(h, [1], scope = 'value', activation = None)
        noise_list.extend(noise)
        return policy, value, noise_list, states, init_state, None, None
    
    
    elif v_net == 'copy':

        if noisy_fc:
            h_p, noise_p = multilayer_fc_noisy(h, input_fc_layers, scope = 'fc_net_p')
            h_v, noise_v = multilayer_fc_noisy(h, input_fc_layers, scope = 'fc_net_v', noise = noise_p)
        else:
            h_p = multilayer_fc(h, input_fc_layers, scope = 'fc_net_p')
            h_v = multilayer_fc(h, input_fc_layers, scope = 'fc_net_v')
            noise_p = noise_v = []
        noise_list.extend(noise_p)
        
        if len(lstm_layers):
            assert input_states_v is not None, 'Specify input states for value network'
            if noisy_lstm:
                lstm_outputs_p, states_p, init_state_p, noise_p = multilayer_lstm_noisy(h_p, M, input_states,
                                                                                        lstm_layers, 'lstm_net_p',
                                                                                        nenvs, nsteps)
                
                lstm_outputs_v, states_v, init_state_v, noise_v = multilayer_lstm_noisy(h_v, M, input_states, 
                                                                                        lstm_layers, 'lstm_net_v',
                                                                                        nenvs, nsteps, noise = noise_p)

            else:
                lstm_outputs_p, states_p, init_state_p = multilayer_lstm(h_p, M, input_states, lstm_layers,
                                                                         'lstm_net_p', nenvs, nsteps)
                
                lstm_outputs_v, states_v, init_state_v = multilayer_lstm(h_v, M, input_states_v, lstm_layers,
                                                                         'lstm_net_v', nenvs, nsteps)
                noise_p = noise_v =  []
            noise_list.extend(noise_p)
            noise_list.extend(noise_v)
            h_p = lstm_outputs_p
            h_v = lstm_outputs_v
        else:
            states_p, init_state_p = None, None
            states_v, init_state_v = None, None

        policy = multilayer_fc(h_p, [num_actions], scope = 'policy', activation = None)
        value = multilayer_fc(h_v, [1], activation = None, scope = 'value')

        return policy, value, noise_list, states_p, init_state_p, states_v, init_state_v
        
        
        
class Network:
    def __init__(self, sess, obs_len, num_actions, nenvs, nsteps, nminibatches = 1, input_fc_layers = [64, 64],
                 lstm_layers = [], output_fc_layers = [], v_net = 'shared', 
                 layer_norm = True, noisy_fc = True, noisy_lstm = True, masked = True):
        self.sess = sess
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.nminibatches = nminibatches
        self.nbatch = (self.nsteps * self.nenvs) // nminibatches
        self.num_actions = num_actions
        
        self.input = tf.placeholder(shape = [self.nbatch, obs_len], name = 'input', dtype = 'float32')
        self.M = tf.placeholder(tf.float32, [self.nbatch])
        self.v_net_type = v_net


        self.noisy_fc = noisy_fc
        self.noisy_lstm = noisy_lstm
        if len(lstm_layers):
            self.input_states = [tf.placeholder(tf.float32, [nenvs//nminibatches, 2*nlstm]) for nlstm in lstm_layers]
            if v_net == 'shared':
                self.input_states_v = None
            elif v_net == 'copy':
                self.input_states_v =  [tf.placeholder(tf.float32, [nenvs//nminibatches, 2*nlstm]) for nlstm in lstm_layers]
        else:
            self.input_states = None
            self.input_states_v = None

        policy, value, noise_list, states, init_state, states_v, init_state_v = build_network(self.input, num_actions, 
                                                                                    self.M, self.input_states,
                                                                                    input_fc_layers, 
                                                                                    lstm_layers,
                                                                                    output_fc_layers,
                                                                                    nenvs = nenvs, 
                                                                                    nsteps = nsteps,
                                                                                    nminibatches = nminibatches,
                                                                                    v_net = v_net,
                                                                                    layer_norm = layer_norm,
                                                                                    noisy_fc = noisy_fc,
                                                                                    noisy_lstm = noisy_lstm,
                                                                                    input_states_v=self.input_states_v)
        self.policy = policy
        self.policy_sm = tf.nn.softmax(policy)
        self.noise_list = noise_list
        self.value = value
        self.states = states
        self.init_state = init_state
        self.states_v = states_v
        self.init_state_v = init_state_v
        self.TEMP = tf.placeholder(tf.float32, [])
        self.LEGAL_MOVES = tf.placeholder(shape = [self.nbatch, self.num_actions], dtype = 'float32')
        self.policy_masked = (policy + self.LEGAL_MOVES)
        self.policy_masked = tf.divide(self.policy_masked, self.TEMP)
        self.policy = tf.divide(self.policy, self.TEMP)
        self.sample_action = tf.squeeze(tf.multinomial(self.policy_masked, 1), axis = 1)
        self.sample_action_oh = tf.one_hot(self.sample_action, depth = num_actions)
        
        if masked:
            self.log_probs = tf.nn.log_softmax(self.policy_masked)
        else:
            self.log_probs = tf.nn.log_softmax(self.policy)
            
        self.logp = tf.reduce_sum(self.sample_action_oh * self.log_probs , axis=1)
        
        if self.states is None:
            self.states = tf.constant([])
            self.states_v = tf.constant([])
    '''def gen_noise(self):
        assert self.noisy, 'Network is not noisy'
        self.generated_noise = []
        for noise_w, noise_b in self.noise_list:
            self.generated_noise.append((np.random.normal(0, 1, size = noise_w.shape), 
                                         np.random.normal(0, 1, size = noise_b.shape)))'''
        
    def step(self, inp, inp_states = None, m = None, inp_states_v = None, legal_moves = None, generated_noise = None,
             temp = 1.):
        if legal_moves is None:
            legal_moves = np.zeros((len(inp), self.num_actions))
            
        feed_dict = {self.input : inp, self.LEGAL_MOVES : legal_moves, self.TEMP : temp}
        
        for noise_ph, noise_val in zip(self.noise_list, generated_noise):
            feed_dict[noise_ph] = noise_val
        

        if inp_states is not None:
            for inp_S, inp_s in zip(self.input_states, inp_states):
                feed_dict[inp_S] = inp_s
            if inp_states_v is not None:
                for inp_S, inp_s in zip(self.input_states_v, inp_states):
                    feed_dict[inp_S] = inp_s
            feed_dict[self.M] = m
            
        if self.states_v is not None:
            actions, probs, logps, values, states, states_v = self.sess.run([self.sample_action, self.log_probs,
                                                                              self.logp, self.value, 
                                                                              self.states, self.states_v], feed_dict)
        else:
            actions, probs, logps, values, states = self.sess.run([self.sample_action, self.log_probs, 
                                                                   self.logp, self.value, 
                                                                   self.states], feed_dict)
            states_v = None
        #print('POLICY', policy)
        #print('AOH', aoh)
        #print('logpros', logprs)
        #print(logps)
        #print(aoh * logprs)
        return actions, probs, logps, values, states, states_v
    
    def get_values(self, inp, inp_states = None, m = None, inp_states_v = None):
        feed_dict = {self.input : inp}
        if inp_states is not None:
            for inp_S, inp_s in zip(self.input_states, inp_states):
                feed_dict[inp_S] = inp_s
            if inp_states_v is not None:
                for inp_S, inp_s in zip(self.input_states_v, inp_states):
                    feed_dict[inp_S] = inp_s
            feed_dict[self.M] = m
            
        values = self.sess.run([self.value], feed_dict)
        
        return values
    
