import numpy as np
import pickle

from .network_building import Network
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
    def __init__(self, num_actions, obs_len, nplayers, nsteps, nenvs,  sess, nminibatches = 1, scope = '', 
                 v_net = 'shared', noisy_fc = True,
                 noisy_lstm = True, gamma = 0.99, fc_input_layers  = [64, 64], lstm_layers = [64], 
                 fc_output_layers = [], layer_norm = True,
                 ent_coef = 0.01, vf_coef = 0.5, masked = True,
                 max_grad_norm = None, lr = 0.001, 
                 epsilon = 1e-5, total_timesteps = int(80e6), 
                 lrschedule='constant', normalize_advs = True):

        #save parameters
        self.scope = scope
        self.nenvs = nenvs
        self.num_actions = num_actions
        self.obs_len = obs_len
        self.nsteps = nsteps
        self.nplayers = nplayers
        self.nminibatches = nminibatches
        self.nbatch = (self.nsteps * self.nenvs * self.nplayers) // nminibatches
        self.lr = lr
        self.train_steps = 0
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advs = normalize_advs
        self.sess = sess
        with tf.variable_scope(scope):
            self.updates = tf.Variable(0, dtype = tf.int32, name = 'total_updates', trainable = False)
            self.increment_updates = self.updates.assign_add(1)
            # create placeholder
            A = tf.placeholder(shape = [self.nbatch], dtype = 'int32', name = 'actions')
            A_OH = tf.one_hot(A, self.num_actions, dtype = 'float32', name = 'actions_oh')
            ADV = tf.placeholder(shape = [self.nbatch], dtype = 'float32', name =  'advantages')
            R = tf.placeholder(shape = [self.nbatch], dtype = 'float32', name = 'R')
            LR = tf.placeholder(tf.float32, [])
            OLDLOGP = tf.placeholder(tf.float32, [None])
            OLDV = tf.placeholder(tf.float32, [None])
            CLIPRANGE = tf.placeholder(tf.float32, [])
            TEMP = tf.placeholder(tf.float32, [])
        with tf.variable_scope(scope + 'ppo_model', reuse=tf.AUTO_REUSE):
            step_model = Network(sess, self.obs_len, self.num_actions, nenvs, 1, 1,
                                 fc_input_layers, lstm_layers, fc_output_layers, v_net,
                                 noisy_fc = noisy_fc, layer_norm = layer_norm, noisy_lstm = noisy_lstm, 
                                 masked = masked)
            
            train_model = Network(sess, self.obs_len, self.num_actions, self.nenvs*self.nplayers, self.nsteps,
                                  nminibatches,
                                  fc_input_layers, lstm_layers, fc_output_layers, v_net,
                                  layer_norm = layer_norm, noisy_fc = noisy_fc, noisy_lstm = noisy_lstm,
                                  masked = masked)
            
        with tf.variable_scope(scope):
            if masked:
                probs = tf.nn.softmax(train_model.policy_masked)
                logp = tf.reduce_sum(A_OH * tf.nn.log_softmax(train_model.policy_masked), axis=1)
            else:
                probs = tf.nn.softmax(train_model.policy)
                logp = tf.reduce_sum(A_OH * tf.nn.log_softmax(train_model.policy), axis=1)


            ratio = tf.exp(logp - OLDLOGP)
            pg_loss1 = ADV * ratio
            pg_loss2 = ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
            entropy =  tf.reduce_mean(tf.reduce_sum(probs * -tf.log(tf.clip_by_value(probs, 1e-7, 1)), axis=1))  
            approx_kl = tf.reduce_mean(OLDLOGP - logp)

            vclipped = OLDV + tf.clip_by_value(train_model.value - OLDV, - CLIPRANGE, CLIPRANGE)
            vf_loss1 = tf.square(train_model.value - R)
            vf_loss2 = tf.square(vclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))


            loss_actor = pg_loss - entropy * ent_coef
            loss_critic = vf_loss * vf_coef
            loss = loss_actor + loss_critic


            params = tf.trainable_variables(scope + 'ppo_model')

            grads_critic = tf.gradients(loss_critic, params)
            if max_grad_norm is not None:
                grads_critic = tf.clip_by_global_norm(grads_critic, max_grad_norm)
            grads_critic = list(zip(grads_critic, params))

            grads_actor = tf.gradients(loss_actor, params)
            if max_grad_norm is not None:
                grads_actor = tf.clip_by_global_norm(grads_actor, max_grad_norm)
            grads_actor = list(zip(grads_actor, params))

            trainer_critic = tf.train.AdamOptimizer(learning_rate = LR, epsilon=epsilon)
            trainer_actor = tf.train.AdamOptimizer(learning_rate = LR, epsilon=epsilon)

            _train_critic = trainer_critic.apply_gradients(grads_critic)
            _train_actor = trainer_actor.apply_gradients(grads_actor)

            self.lr = Scheduler(v = lr, nvalues = total_timesteps, schedule = lrschedule)
            
        def change_lr(new_lr):
             self.lr = Scheduler(v = new_lr, nvalues = total_timesteps, schedule = lrschedule)
        self.change_lr = change_lr 
          
            
        def train(obs, cliprange, neglogps, inp_states, rewards, 
                  masks, actions, values, inp_states_v, legal_moves, generated_noise, temp = 1.):    
            advs = rewards - values
            if normalize_advs:
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            feed_dict = {train_model.input:obs, A:actions, ADV:advs,
                         R:rewards, LR:cur_lr, train_model.LEGAL_MOVES : legal_moves,
                         OLDV : values, OLDLOGP : neglogps, CLIPRANGE : cliprange, train_model.TEMP : temp}
            for noise_ph, noise_val in zip(train_model.noise_list, generated_noise):
                feed_dict[noise_ph] = noise_val
            if inp_states is not None:
                for inp_S, inp_s in zip(self.train_model.input_states, inp_states):
                    feed_dict[inp_S] = inp_s
                if inp_states_v is not None and self.train_model.input_states_v:
                    for inp_S, inp_s in zip(self.train_model.input_states_v, inp_states_v):
                        feed_dict[inp_S] = inp_s
                feed_dict[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, kl, probs_val, _, _ = sess.run([pg_loss, vf_loss, 
                                                                                     entropy, approx_kl, probs,
                                                                                     _train_actor, _train_critic],
                                                                                    feed_dict)
            
            self.train_steps += 1
            return policy_loss, value_loss, policy_entropy, kl, probs_val


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.get_values
        self.init_state = step_model.init_state
        self.init_state_v = step_model.init_state_v
        tf.global_variables_initializer().run(session = sess)
   
