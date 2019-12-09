import time
import numpy as np
import os

from hanabi_learning_environment import rl_env
from tf_agents_lib.pyhanabi_env_wrapper import *

from PPOAgentDynamic.util import *
from PPOAgentDynamic.Model import Model
from PPOAgentDynamic.Game import Game
from PPOAgentDynamic.SPGame import SPGame
import tensorflow as tf

class MiniModel():
    def __init__(self, nobs, nactions, scope, sess, hidden_units = 32, tr_method = 'hidden'):
        self.sess = sess
        self.nactions = nactions
        self.tr_method = tr_method
        if tr_method is 'hidden':
            self.tr_shape = hidden_units
        elif tr_method is 'probs':
            self.tr_shape = nactions
        with tf.variable_scope(scope, reuse = False):
            self.INP = tf.placeholder(dtype = 'float32', shape = [None, nobs])
            self.LM = tf.placeholder(shape = [None, nactions], dtype = 'float32')
            self.A = tf.placeholder(shape = [None], dtype = 'int32', name = 'actions')
            self.A_OH = tf.one_hot(self.A, nactions, dtype = 'float32', name = 'actions_oh')
            if hidden_units is not None and hidden_units > 0 :
                h = multilayer_fc(self.INP, [hidden_units], scope = 'fc_net', layer_norm = False)
                self.hidden = h
            else:
                h = self.INP
            
            self.output = multilayer_fc(h, [nactions], scope = 'output', layer_norm = False,
                                   activation = None,) + self.LM
            self.output_probs = tf.nn.log_softmax(self.output)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.A_OH, 
                                                                                logits = self.output_probs))
            params = tf.trainable_variables(scope)
            
            grads = tf.gradients(self.loss, params)
            grads = list(zip(grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate = 1e-3, )
            self._train = trainer.apply_gradients(grads)
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.sess.run(tf.variables_initializer(variables))
        
    
    def train(self, nepochs, inp_train, actions_train, lm_train):
        feed_dict = {self.INP : inp_train, self.A : actions_train, self.LM : lm_train}
        loss_hist = []
        for i in range(nepochs):
            loss_eval, _, output_probs_eval  = self.sess.run([self.loss, self._train, self.output_probs], 
                                                        feed_dict = feed_dict)
            loss_hist.append(loss_eval)
        return np.mean(loss_hist)
    
    def sample_actions(self, inp, actions, lm):
        feed_dict = {self.INP : inp, self.A : actions, self.LM : lm}
        
        loss_eval, _, output_probs_eval  = self.sess.run([self.loss, self._train, self.output_probs],
                                                    feed_dict = feed_dict)
        actions_sim = []
        for p in output_probs_eval:
            actions_sim.append(np.random.choice(np.arange(0, self.nactions), p = np.exp(p)))
        actions_sim = np.array(actions_sim)
        return actions_sim

    '''def transform(self, inp):
        feed_dict = {self.INP : inp}
        inp_trans = self.sess.run([self.transformed_input], feed_dict = feed_dict)
        return inp_trans[0]'''
    
    def transform(self, inp, lm):
        # how: 'probs', 'hidden', 'weights'
        feed_dict = {self.INP : inp, self.LM : lm}
        if self.tr_method is 'probs':
            inp_trans = self.sess.run([self.output_probs], feed_dict = feed_dict)
        elif self.tr_method is 'hidden':
            inp_trans = self.sess.run([self.hidden], feed_dict = feed_dict)
        elif self.tr_method is 'weights':
            inp_trans = [[]]
        else:
            print('Wrong method of transformation!')
            raise 'MethodError'
        return inp_trans[0]
    
    
def load_hanabi(ENV_CONFIG):
    # loads wrapped env
    return PyhanabiEnvWrapper(rl_env.make(**ENV_CONFIG))

def clip(rewards, min_val, max_val):
    return list(map(lambda x: max(min_val, min(x, max_val)), rewards))

def learn(run_name, nupdates, k, ENV_CONFIG, MODEL_CONFIG,  REWARDS_CONFIG, num_episodes = 90, 
          wait_rewards = True, single_player = False,
          target_kl_init = 0.5, target_kl_goal = 0.01, kl_factor = 0.995,
          root_folder = './experiments/', load = False, write_sum = True, eval_every = 100):
    
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    load_env = lambda: load_hanabi(ENV_CONFIG)
    env = load_hanabi(ENV_CONFIG)
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    nactions = action_spec.maximum + 1 - action_spec.minimum
    nobs = obs_spec['state'].shape[0]
    print('NOBS:', nobs)
    nplayers = ENV_CONFIG['num_players']
    
    model = Model(nactions, nobs, nplayers, sess = sess,
                     **MODEL_CONFIG)
    if single_player:
        game = SPGame(model, load_env, REWARDS_CONFIG)
    else:
        game = Game(nplayers, model, load_env, REWARDS_CONFIG, wait_rewards = True)
    
    path = root_folder + ENV_CONFIG['environment_name'] + '-' + str(ENV_CONFIG['num_players'])
    if not os.path.isdir(path):
        os.makedirs(path)

    summary_writer = tf.summary.FileWriter(path + '/summary/' + run_name)
    saver = tf.train.Saver(max_to_keep=5)
    
    loss_history, rewards_history, scores_history, lengths_history = [], [], [], []
    custom_rewards_history = {'hint_reward' : [],
                           'play_reward' : [],
                           'discard_reward' : [] }
    training_steps_history = []
    target_kl = target_kl_init
    if not os.path.isdir(path + '/models/' + run_name):
        os.makedirs(path + '/models/' + run_name)
    if load:
        ckpt = tf.train.get_checkpoint_state(path + '/models/' + run_name)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    init_num_upd = sess.run(model.updates)
    for nupd in range(0, nupdates):
        
        time_start = time.time()
        steps_start = game.total_steps
        score, rewards, lengths, custom_rewards = game.play_untill_train(num_episodes)
        for key in custom_rewards_history:
            custom_rewards_history[key].extend(custom_rewards[key])
        scores_history.append(np.mean(score))
        rewards_history.append(np.mean(rewards))
        lengths_history.append(np.mean(lengths))
        
        state_probs, policy_loss, value_loss, policy_entropy, i = game.train_model(k, target_kl)
        training_steps_history.append(i)
                                       
        target_kl = max(target_kl_goal, kl_factor * target_kl)

        loss_history.append((policy_loss, value_loss, policy_entropy))
        time_end = time.time()
        steps_end =  game.total_steps
        
        if nupd % 50 == 0:
            
            print('-----------------UPDATE%d----------------' % (nupd + init_num_upd))
            print('R:%.2f' % np.mean(rewards_history), 
                  'Score: %.2f' % np.mean(scores_history),
                  'Length: %d' % np.mean(lengths_history),
                  'Updates/batch: %d' % int(np.mean(training_steps_history)))
            print('%.1f steps/second' % ((steps_end-steps_start)/(time_end-time_start)) )
            print('%d steps in total' % steps_end)
            print('Current learning rate is %.5f' % model.lr.value())
            for key in custom_rewards_history:
                print(key, np.mean(custom_rewards_history[key]))
            print('-------------------------------------------')
            saver.save(sess, path + '/models/' + run_name + '/model-' + str(init_num_upd + nupd) + '.cptk')
            average_training_steps = []
            custom_rewards_history = {key : [] for key in custom_rewards_history}

                  
        if write_sum:
            if nupd % 10 == 0 and nupd > 0:
                loss_history = np.array(loss_history)
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', 
                                  simple_value = float(np.mean(rewards_history)))
                summary.value.add(tag='Perf/Score', 
                                  simple_value = float(np.mean(scores_history)))
                summary.value.add(tag='Perf/EpLength', 
                                  simple_value = float(np.mean(lengths_history)))
                summary.value.add(tag='Perf/Training steps per epoch', 
                                  simple_value = float(np.mean(training_steps_history)))
                summary.value.add(tag='Losses/Policy Loss', 
                                  simple_value = float(np.mean(loss_history[:, 0])))
                summary.value.add(tag='Losses/Value Loss',
                                  simple_value = float(np.mean(loss_history[:, 1])))
                summary.value.add(tag='Losses/Entropy', 
                                  simple_value = float(np.mean(loss_history[:, 2])))
                
                summary_writer.add_summary(summary, nupd + init_num_upd)
                summary_writer.flush()
                
                loss_history, rewards_history, scores_history, lengths_history = [], [], [], []
                training_steps_history = []
    return game, model, sess