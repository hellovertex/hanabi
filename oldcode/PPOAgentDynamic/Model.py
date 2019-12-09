import functools
import numpy as np
import tensorflow as tf
from .network_building import *
from baselines.a2c.utils import Scheduler
from baselines.common import tf_util
from tf_agents.utils import tensor_normalizer as tens_norm
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
# tf_agents to this scirpt dictionary:
# importance_ratio_clipping : 1 - CLIPRANGE
class Model(object):
    def __init__(self, num_actions, obs_len, nplayers, nenvs,  sess, scope = '', v_net = 'shared', noisy_fc = True,
                 gamma = 0.99, fc_input_layers  = [64, 64], layer_norm = True,
                 ent_coef = 0.01, vf_coef = 0.5, masked = True, 
                 normalize_advs = True, normalize_rewards = True, r_clip = 10,
                 max_grad_norm = None, lr = 0.001, 
                 epsilon = 1e-5, total_timesteps = int(80e6), lrschedule = 'constant', 
                 use_kl_penalty = False, use_clipping = True,
                 initial_adaptive_kl_beta = 1.0, adaptive_kl_target = 0.01):
        

        #save parameters
        self.scope = scope
        self.nenvs = nenvs
        self.num_actions = num_actions
        self.obs_len = obs_len
        self.nplayers = nplayers
        self.lr = lr
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advs = normalize_advs
        self.sess = sess
        self.nbatch = None
        self.use_kl_penalty = use_kl_penalty
        
        with tf.variable_scope(scope):
            self.updates = tf.Variable(0, dtype = tf.int32, name = 'total_updates', trainable = False)
            self.increment_updates = self.updates.assign_add(1)
            
            # create placeholder
            A = tf.placeholder(shape = [self.nbatch], dtype = 'int32', name = 'actions')
            A_OH = tf.one_hot(A, self.num_actions, dtype = 'float32', name = 'actions_oh')
            ADV = tf.placeholder(shape = [self.nbatch], dtype = 'float32', name =  'advantages')
            R = tf.placeholder(shape = [self.nbatch], dtype = 'float32', name = 'R')
            if normalize_rewards:
                reward_normalizer = tens_norm.StreamingTensorNormalizer(tensor_spec.TensorSpec([],tf.float32))
                R_NORM = reward_normalizer.normalize(R, center_mean = False, clip_value = r_clip)
            LR = tf.placeholder(tf.float32, [])
            OLDLOGP = tf.placeholder(tf.float32, [None])
            OLDPROBS = tf.placeholder(tf.float32, [None, self.num_actions])
            OLDV = tf.placeholder(tf.float32, [None])
            CLIPRANGE = tf.placeholder(tf.float32, [])
            TEMP = tf.placeholder(tf.float32, [])
            
        with tf.variable_scope(scope + 'ppo_model', reuse=tf.AUTO_REUSE):
            model = Network(sess, self.obs_len, self.num_actions, nenvs,
                            fc_input_layers, v_net, noisy_fc = noisy_fc,
                            layer_norm = layer_norm, masked = masked)
        
    #kl_cutoff = self._kl_cutoff_factor * self._adaptive_kl_target
    #mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    #kl_over_cutoff = tf.maximum(mean_kl - kl_cutoff, 0.0)
    #kl_cutoff_loss = self._kl_cutoff_coef * tf.square(kl_over_cutoff)
        with tf.variable_scope(scope):
            if masked:
                probs = tf.nn.softmax(model.policy_masked)
                logp = tf.reduce_sum(A_OH * tf.nn.log_softmax(model.policy_masked), axis=1)
            else:
                probs = tf.nn.softmax(model.policy)
                logp = tf.reduce_sum(A_OH * tf.nn.log_softmax(model.policy), axis=1)


            ratio = tf.exp(logp - OLDLOGP)
            pg_loss1 = ADV * ratio
            pg_loss2 = ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
            entropy =  tf.reduce_mean(tf.reduce_sum(probs * -tf.log(tf.clip_by_value(probs, 1e-7, 1)), axis=1))  
            approx_kl = tf.reduce_mean(OLDLOGP - logp)

            vclipped = OLDV + tf.clip_by_value(model.value - OLDV, - CLIPRANGE, CLIPRANGE)
            if normalize_rewards:
                vf_loss1 = tf.square(model.value - R_NORM)
                vf_loss2 = tf.square(vclipped - R_NORM)
            else:
                vf_loss1 = tf.square(model.value - R)
                vf_loss2 = tf.square(vclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
            loss_actor = pg_loss - entropy * ent_coef
            loss_critic = vf_loss * vf_coef
            
            # kl penalty
            newdist =  probs + 1e-10
            olddist = tf.exp(OLDPROBS) + 1e-10
            kl_divergence = olddist * tf.log(olddist/newdist)
            mean_kl = tf.reduce_mean(input_tensor = kl_divergence)
            adaptive_kl_beta = common.create_variable('adaptive_kl_beta', initial_adaptive_kl_beta, 
                                                      dtype = tf.float32)
            kl_loss = adaptive_kl_beta * mean_kl
            if self.use_kl_penalty:
                loss_actor += kl_loss
                # adaptive update op
                factor = tf.case([(mean_kl <= adaptive_kl_target / 1.5, lambda: tf.constant(0.5, dtype=tf.float32)),
                                  (mean_kl >  adaptive_kl_target * 1.5, lambda: tf.constant(2.0, dtype=tf.float32)),
                                 ], default=lambda: tf.constant(1.0, dtype=tf.float32), exclusive=True)
                beta_update_op = tf.compat.v1.assign(adaptive_kl_beta, adaptive_kl_beta * factor)
                
            loss = loss_actor + loss_critic 
            params = tf.trainable_variables(scope + 'ppo_model')

            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate = LR, epsilon=epsilon)
            _train = trainer.apply_gradients(grads)

            self.lr = Scheduler(v = lr, nvalues = total_timesteps, schedule = lrschedule)
            
        def change_lr(new_lr):
             self.lr = Scheduler(v = new_lr, nvalues = total_timesteps, schedule = lrschedule)
        def update_lr():
            _ = self.lr.value()
            
        self.change_lr = change_lr 
        self.update_lr = update_lr

        def train(obs, cliprange, neglogps, oldprobs, rewards, 
                  actions, values, legal_moves, generated_noise, update_kl = False):
            
            advs = rewards - values
            if normalize_advs:
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            feed_dict = {model.input:obs, A:actions, ADV:advs,
                         R:rewards, LR:cur_lr, model.LEGAL_MOVES : legal_moves,
                         OLDV : values, OLDLOGP : neglogps, OLDPROBS: oldprobs, CLIPRANGE : cliprange}
            
            for noise_ph, noise_val in zip(model.noise_list, generated_noise):
                feed_dict[noise_ph] = noise_val
            
            odval, ndval, mean_kl_val, kl_loss_val, beta = sess.run([olddist, newdist, mean_kl, kl_loss, 
                                                                     adaptive_kl_beta], feed_dict)
            values_to_estimate = [pg_loss, vf_loss, entropy, kl_loss, probs]
            updates = [_train]
            policy_loss, value_loss, policy_entropy, kl, probs_val, _, = sess.run(values_to_estimate + 
                                                                                  updates, feed_dict)
                
            #if update_kl:
            #    _, new_kl_beta = sess.run([beta_update_op, adaptive_kl_beta], feed_dict)
                #print('NEW', new_kl_beta)
                

            return policy_loss, value_loss, policy_entropy, kl, probs_val
    
        self.train = train
        self.model = model
        self.step = model.step
        self.total_steps = model.total_steps
        self.increment_steps = model.increment_steps
        self.value = model.get_values
        tf.global_variables_initializer().run(session = sess)
   