import numpy as np
import tensorflow as tf
from scipy.special import softmax
from .util import *

def build_network(observation, num_actions, input_fc_layers = [256, 256], 
                  nenvs = 8, v_net = 'shared', noisy_fc = False, layer_norm = True):
    
    noise_list = []
    h = observation
    if v_net == 'shared':
        if noisy_fc:
            h, noise = multilayer_fc_noisy(h, input_fc_layers, scope = 'fc_net', layer_norm = layer_norm)
        else:
            h = multilayer_fc(h, input_fc_layers, scope = 'fc_net', layer_norm = layer_norm)
            noise = []
        noise_list.extend(noise)
        

        policy = multilayer_fc(h, [num_actions], scope = 'policy', activation = None)
        value = multilayer_fc(h, [1], scope = 'value', activation = None)
        noise_list.extend(noise)
    
    
    elif v_net == 'copy':

        if noisy_fc:
            h_p, noise_p = multilayer_fc_noisy(h, input_fc_layers, scope = 'fc_net_p')
            h_v, noise_v = multilayer_fc_noisy(h, input_fc_layers, scope = 'fc_net_v', noise = noise_p)
        else:
            h_p = multilayer_fc(h, input_fc_layers, scope = 'fc_net_p')
            h_v = multilayer_fc(h, input_fc_layers, scope = 'fc_net_v')
            noise_p = noise_v = []
        noise_list.extend(noise_p)
        noise_list.extend(noise_v)


        policy = multilayer_fc(h_p, [num_actions], scope = 'policy', activation = None)
        value = multilayer_fc(h_v, [1], activation = None, scope = 'value')

    return policy, value, noise_list
        
        
        
class Network:
    def __init__(self, sess, obs_len, num_actions, nenvs, input_fc_layers = [64, 64],
                 v_net = 'shared', layer_norm = True, noisy_fc = True, masked = True):
        
        self.sess = sess
        self.nenvs = nenvs
        self.num_actions = num_actions
        
        self.input = tf.placeholder(shape = [None, obs_len], name = 'input', dtype = 'float32')
        self.v_net_type = v_net
        self.noisy_fc = noisy_fc

        policy, value, noise_list = build_network(self.input, num_actions, input_fc_layers,
                                                  nenvs = nenvs, v_net = v_net,
                                                  layer_norm = layer_norm, noisy_fc = noisy_fc)
                                                                                    
        self.total_steps = tf.Variable(0, dtype = tf.int32, name = 'total_steps', trainable = False)
        self.increment_steps = self.total_steps.assign_add(self.nenvs)
        self.policy = policy
        self.policy_sm = tf.nn.softmax(policy)
        
        self.LEGAL_MOVES = tf.placeholder(shape = [None, self.num_actions], dtype = 'float32')
        self.policy_masked = (policy + self.LEGAL_MOVES)
        
        self.value = value
        self.noise_list = noise_list
        
        self.sample_action = tf.squeeze(tf.multinomial(self.policy_masked, 1), axis = 1)
        self.sample_action_oh = tf.one_hot(self.sample_action, depth = num_actions)
        
        if masked:
            self.log_probs = tf.nn.log_softmax(self.policy_masked)
        else:
            self.log_probs = tf.nn.log_softmax(self.policy)
            
        self.logp = tf.reduce_sum(self.sample_action_oh * self.log_probs , axis=1)

        
    def step(self, inp, legal_moves = None, generated_noise = None):
        if legal_moves is None:
            legal_moves = np.zeros((len(inp), self.num_actions))
            
        feed_dict = {self.input : inp, self.LEGAL_MOVES : legal_moves}
        
        for noise_ph, noise_val in zip(self.noise_list, generated_noise):
            feed_dict[noise_ph] = noise_val
        
            
        actions, probs, logps, values, _ = self.sess.run([self.sample_action, self.log_probs, self.logp, self.value, 
                                                          self.increment_steps], feed_dict)

        return actions, probs, logps, values
    
    def get_values(self, inp, generated_noise = None):
        
        feed_dict = {self.input : inp}    
        for noise_ph, noise_val in zip(self.noise_list, generated_noise):
            feed_dict[noise_ph] = noise_val
        
        
        values = self.sess.run([self.value], feed_dict)
        
        return values
    
