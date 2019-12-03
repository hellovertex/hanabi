import numpy as np
import pickle

from .network import Network
from .util import Scheduler

import tensorflow as tf
from tf_agents.utils import tensor_normalizer as tens_norm
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


REWARDS_KEYS = ['baseline', 'play0', 'play1', 'play2', 'play3', 'play4',
                'discard_last_copy', 'discard_extra',
                'hint_last_copy', 'hint_penalty',  'hint_playable',
                'use_hamming', 'loose_life']


class Model(object):
    def __init__(self, nactions, nobs, nplayers, nenvs,  sess, scope = '', nsteps = None,
                 fc_input_layers  = [64, 64], v_net = 'shared', noisy_fc = True, layer_norm = True,
                 gamma = 0.99, ent_coef = 0.01, vf_coef = 0.5, cliprange = 0.2, rewards_config = {},
                 max_grad_norm = None, k = 8,  lr = 0.001, lr_half_period = int(1e6), anneal_lr = True,
                 normalize_advs = True, epsilon = 1e-5, path = './experiments/PBT/'):

        #save parameters
        self.type = 'dynamic' 
        self.scope = scope
        self.nenvs = nenvs
        self.nactions = nactions
        self.nobs = nobs
        self.nplayers = nplayers
        self.lr = lr
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advs = normalize_advs
        self.sess = sess
        self.path = path + scope + '/'
        self.change_rewards_config(rewards_config)
        self.anneal_lr = anneal_lr
        
        with tf.variable_scope(scope):
            # CREATE VARIABLES AFFECTING TRAINING
            self.lr = tf.Variable(lr, dtype = tf.float32, name = 'learning_rate', trainable = False)
            self.k = tf.Variable(k, dtype = tf.int32, name = 'k', trainable = False)
            if nsteps is None:
                nsteps_init = -1
            else:
                nsteps_init = nsteps
            self.nsteps = tf.Variable(nsteps_init, dtype = tf.int32, name = 'nsteps', trainable = False)
            self.cliprange = tf.Variable(cliprange, dtype = tf.float32, name = 'cliprange', trainable = False)
            self.vf_coef = tf.Variable(vf_coef, dtype = tf.float32, name = 'value_loss_coef', trainable = False)
            self.ent_coef = tf.Variable(ent_coef, dtype = tf.float32, name = 'entropy_loss_coef', trainable = False)
            self.train_epochs = tf.Variable(0, dtype = tf.int32, name = 'total_updates', trainable = False)
            self.timesteps = tf.Variable(0, dtype = tf.int32, name = 'timesteps', trainable = False)
            # ops for updating variables
            self.update_lr = tf.assign(self.lr, self.lr * np.power(0.5, 1 / lr_half_period))
            self.update_train_epochs = self.train_epochs.assign_add(1)
            self.update_timesteps = self.timesteps.assign_add(nenvs)
            # CREATE PLACEHOLDERS
            
            self.OBS = tf.placeholder(shape = [None, nobs], dtype = 'float32', name = 'observations')
            self.LEGAL_MOVES = tf.placeholder(shape = [None, self.nactions], dtype = 'float32', name = 'legal_moves')
            self.A = tf.placeholder(shape = [None, ], dtype = 'int32', name = 'actions')
            self.A_OH = tf.one_hot(self.A, self.nactions, dtype = 'float32', name = 'actions_oh')
            self.ADVS = tf.placeholder(shape = [None, ], dtype = 'float32', name =  'advantages')
            self.R = tf.placeholder(shape = [None, ], dtype = 'float32', name = 'returns')
            self.OLD_ALOGP = tf.placeholder(tf.float32, [None], name = 'action_log_old')
            self.OLD_PROBS = tf.placeholder(tf.float32, [None, self.nactions], name = 'probs_old')
            self.OLD_VALUE = tf.placeholder(tf.float32, [None], name = 'values_old')
   
            with tf.variable_scope('network', reuse = tf.AUTO_REUSE):
            # CREATE NETWORK
                network = Network(self.OBS, self.LEGAL_MOVES, self.nactions,
                                  fc_input_layers, v_net, noisy_fc, layer_norm)

            # DEFINE LOSSES                         
            # POLICY LOSS
            new_alogp = tf.reduce_sum(self.A_OH * network.logp, axis=1)
            ratio = tf.exp(new_alogp - self.OLD_ALOGP)
            ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.pg_loss = -tf.reduce_mean(tf.minimum(self.ADVS * ratio,
                                                      self.ADVS * ratio_clipped))
            # ENTROPY LOSS
            self.entropy =  tf.reduce_mean(tf.reduce_sum(- network.probs * 
                                                         tf.log(tf.clip_by_value(network.probs, 1e-7, 1)), axis = 1))
            # VALUE LOSS
            value_clipped = self.OLD_VALUE + tf.clip_by_value(network.value - self.OLD_VALUE, 
                                                              - self.cliprange, self.cliprange)                
            self.vf_loss = .5 * tf.reduce_mean(tf.maximum(tf.square(network.value - self.R), 
                                                          tf.square(value_clipped - self.R)))
            # LOSSES FOR ACTOR AND CRITIC NETWORKS
            self.loss_actor = self.pg_loss - self.ent_coef * self.entropy
            self.loss_critic = self.vf_coef * self.vf_loss 
            # TOTAL LOSS
            self.loss = self.loss_actor + self.loss_critic 
            
            # GRADIENTS, OPTIMIZER, TRAIN OP
            params = tf.trainable_variables(scope + '/network')
            grads_actor = tf.gradients(self.loss_actor, params)
            grads_critic = tf.gradients(self.loss_critic, params)
            if max_grad_norm is not None:
                grads_actor = tf.clip_by_global_norm(grads_actor, max_grad_norm)
                grads_critic = tf.clip_by_global_norm(grads_critic, max_grad_norm)
                
            grads_actor = list(zip(grads_actor, params))
            grads_critic = list(zip(grads_critic, params))
            trainer_actor = tf.train.AdamOptimizer(learning_rate = self.lr, epsilon = epsilon)
            trainer_critic = tf.train.AdamOptimizer(learning_rate = self.lr, epsilon = epsilon)
            # one op for botch critic and actor
            self._train_actor = trainer_actor.apply_gradients(grads_actor)
            self._train_critic= trainer_critic.apply_gradients(grads_critic)
            
            
        self.saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope))
        self.writer = tf.summary.FileWriter(path + scope + '/summary/' )
        self.network = network

        sess.run(tf.global_variables_initializer())
        
    def step(self, obs, legal_moves, noise = None):
        # feed values
        feed_dict = {self.OBS : obs, self.LEGAL_MOVES : legal_moves}
        for noise_ph, noise_val in zip(self.network.noise_list, noise):
            feed_dict[noise_ph] = noise_val
        
        actions, probs, alogps, values, _ = self.sess.run([self.network.sample_action, self.network.probs, 
                                                           self.network.alogp, self.network.value, 
                                                           self.update_timesteps], feed_dict)  
        return actions, probs, alogps, values
                                       
    def train(self, obs, actions, probs, alogp, legal_moves, values, returns, noise = None):
        # estimate advantages
        advs = returns - values
        if self.normalize_advs:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # feed values
        feed_dict = {self.OBS : obs, self.A : actions, self.ADVS : advs,
                     self.R : returns, self.LEGAL_MOVES : legal_moves,
                     self.OLD_VALUE : values, self.OLD_ALOGP : alogp, 
                     self.OLD_PROBS: probs}
            # feed values of noise for noisy net
        for noise_ph, noise_val in zip(self.network.noise_list, noise):
            feed_dict[noise_ph] = noise_val
                
        # get current k, update train epochs
        k_val = self.sess.run(self.k)
        self.sess.run(self.update_train_epochs)
            
        # optimize
        mean_ploss_val, mean_vloss_val, mean_entropy_val = 0, 0, 0
        for nupd in range(k_val):
            ploss_val, vloss_val, entropy_val, _, _ = self.sess.run([self.pg_loss, self.vf_loss, self.entropy,
                                                                     self._train_actor, self._train_critic], 
                                                                    feed_dict)
            mean_ploss_val += ploss_val
            mean_vloss_val += vloss_val
            mean_entropy_val += entropy_val
               
        if self.anneal_lr:
            self.sess.run([self.update_lr])
            
            
        return mean_ploss_val / k_val, mean_vloss_val / k_val, mean_entropy_val / k_val


    def save_model(self,):
        self.saver.save(self.sess, self.path + 'model/model.cptk')
        with open(self.path + '/rewards_dict.pkl', 'wb') as f:
            pickle.dump(self.rewards_config, f)
            
    def load_model(self, new_lr = False):
        ckpt = tf.train.get_checkpoint_state(self.path + 'model/')
        if ckpt is None:
            print('Could not load model "%s" at %s' % (self.scope, self.path + 'model/'))
        else:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            with open(self.path + 'rewards_dict.pkl', 'rb') as f:
                self.rewards_config = pickle.load(f)
            if new_lr:
                self.sess.run([self.lr.assign(new_lr)])
            train_epochs, ts, lr, k, cliprange = self.sess.run([self.train_epochs, self.timesteps,
                                                                 self.lr, self.k, self.cliprange])
            print('Successfully loaded model "%s":' % self.scope)
            print('  "%s" was trained for %d epochs, using %d timesteps' %(self.scope, train_epochs, ts))
            print('  "%s" has following parameters: k = %d, cliprange = %.1f, lr = %f' % (self.scope, k, cliprange, lr))
            print('  "%s" has following rewards_config:' % (self.scope), self.rewards_config)
            
    def save_params_summary(self,):
        summary = tf.Summary()
        summary.value.add(tag = 'Params/Learning Rate', 
                          simple_value = self.sess.run(self.lr))
        summary.value.add(tag = 'Params/K', 
                          simple_value = self.sess.run(self.k))
        summary.value.add(tag = 'Params/Cliprange', 
                          simple_value = self.sess.run(self.cliprange))
        summary.value.add(tag = 'Params/Vf_coef', 
                          simple_value = self.sess.run(self.vf_coef))
        summary.value.add(tag = 'Params/Ent_coef', 
                          simple_value = self.sess.run(self.ent_coef))
        self.writer.add_summary(summary, self.sess.run(self.train_epochs))
        self.writer.flush()

    def change_rewards_config(self, new_rewards_config):
        self.rewards_config = new_rewards_config
        
    def save_rewards_config_summary(self, ):
        summary = tf.Summary()
        for key in self.rewards_config:
            summary.value.add(tag = 'Rewards/' + key, simple_value = self.rewards_config[key])
        self.writer.add_summary(summary, self.sess.run(self.train_epochs))
        self.writer.flush()
        

   