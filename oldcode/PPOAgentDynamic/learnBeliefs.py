import time
import rl_env
import numpy as np
import os
from PPOAgent.util import *
from PyHanabiWrapper import *
from PPOAgent.Model import Model
from PPOAgent.GameBeliefs import Game
import tensorflow as tf
from CardCounting import *

class MiniModel():
    def __init__(self, nobs, nactions, scope, sess, hidden_units = 32):
        self.sess = sess
        self.nactions = nactions
        with tf.variable_scope(scope, reuse = False):
            self.INP = tf.placeholder(dtype = 'float32', shape = [None, nobs])
            self.LM = tf.placeholder(shape = [None, nactions], dtype = 'float32')
            self.A = tf.placeholder(shape = [None], dtype = 'int32', name = 'actions')
            self.A_OH = tf.one_hot(self.A, nactions, dtype = 'float32', name = 'actions_oh')
            if hidden_units is not None and hidden_units > 0 :
                h = multilayer_fc(self.INP, [hidden_units], scope = 'fc_net', layer_norm = False)
            else:
                h = self.INP
            
            self.output = multilayer_fc(h, [nactions], scope = 'output', layer_norm = False,
                                   activation = None,)
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

    
    def compute_action_probs(self, hands, inp, lm):
        # how: 'probs', 'hidden', 'weights'
        
        new_input = []
        for h in hands:
            new_input.append(np.concatenate([h, inp[OBSERVED_HANDS_END:]], 0))
        feed_dict = {self.INP : np.array(new_input)}
        probs = self.sess.run([self.output_probs], feed_dict = feed_dict)
        return probs[0]
    
    
def load_hanabi(ENV_CONFIG):
    # loads wrapped env
    return PyhanabiEnvWrapper(rl_env.make(**ENV_CONFIG))

def clip(rewards, min_val, max_val):
    return list(map(lambda x: max(min_val, min(x, max_val)), rewards))

def learnBeliefs(run_name, nupdates, k, ENV_CONFIG, MODEL_CONFIG, clip_func,
                 mini_model_units = 32, train_mini_model = 10,
                 target_kl_init = 0.5, target_kl_goal = 0.01, kl_factor = 0.995,
                 root_folder = './experiments/', load = False, write_sum = True, eval_every = 100):
    
    
    tf.reset_default_graph()
    sess = tf.Session()

    (NUM_RANKS, NUM_COLORS, COLORS_ORDER, HAND_SIZE,
     BITS_PER_CARD, INITIAL_CARD_COUNT, TOTAL_CARDS, HINT_MASK) = set_game_vars(ENV_CONFIG['environment_name'],
                                                                                ENV_CONFIG['num_players'])
    set_indexes()
    
    load_env = lambda: load_hanabi(ENV_CONFIG)
    env = load_hanabi(ENV_CONFIG)
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    nactions = action_spec.maximum + 1 - action_spec.minimum
    nobs = obs_spec['state'].shape[0]
    nplayers = ENV_CONFIG['num_players']
    
    
    mini_model = MiniModel(nobs, nactions, MODEL_CONFIG['scope'] + '/MM', sess, 
                               mini_model_units)
    nobs_ext = nobs + HAND_SIZE * BITS_PER_CARD
    model = Model(nactions, nobs_ext, nplayers, sess = sess,
                     **MODEL_CONFIG)
    game = Game(nplayers, model, mini_model, load_env, clip_func = clip_func)
    
    path = root_folder + ENV_CONFIG['environment_name'] + '-' + str(ENV_CONFIG['num_players'])
    if not os.path.isdir(path):
        os.makedirs(path)

    summary_writer = tf.summary.FileWriter(path + '/summary/' + run_name)
    saver = tf.train.Saver(max_to_keep=5)
    
    loss_history, rewards_history, clipped_rewards_history, lengths_history = [], [], [], []
    training_steps_history = []
    mm_loss_hist = []
    target_kl = target_kl_init
    if not os.path.isdir(path + '/models/' + run_name):
        os.makedirs(path + '/models/' + run_name)
    if load:
        ckpt = tf.train.get_checkpoint_state(path + '/models/' + run_name)
        saver.restore(sess,ckpt. model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    init_num_upd = sess.run(model.updates)
    for nupd in range(0, nupdates):
        time_start = time.time()
        steps_start = game.total_steps
        
        clipped_rewards, rewards, lengths = game.play_untill_train()
        clipped_rewards_history.append(np.mean(clipped_rewards))
        rewards_history.append(np.mean(rewards))
        lengths_history.append(np.mean(lengths))
        
        state_probs, policy_loss, value_loss, policy_entropy, i = game.train_model(k, target_kl)
        training_steps_history.append(i)
        inps, actions, probs, lms = state_probs
        if nupd % 50 == 0:
            mm_actions = mini_model.sample_actions(inps, actions, lms)
            last_mm_acc = np.mean(mm_actions == actions)
            
        mm_loss = mini_model.train(max(train_mini_model, i), inps, actions, lms)
        mm_loss_hist.append(mm_loss)
                                       
        target_kl = max(target_kl_goal, kl_factor * target_kl)

        loss_history.append((policy_loss, value_loss, policy_entropy))
        time_end = time.time()
        steps_end =  game.total_steps
        wasted_steps = steps_end - steps_start - model.nbatch
        wasted_percenteage = wasted_steps / (steps_end - steps_start)
        
        if nupd % 5 == 0:
            
            print('-----------------UPDATE%d----------------' % (nupd + init_num_upd))
            print('R:%.2f' % np.mean(rewards_history), 
                  'R_clipped: %.2f' % np.mean(clipped_rewards_history),
                  'Length: %d' % np.mean(lengths_history),
                  'Updates/batch: %d' % int(np.mean(training_steps_history)))
            print('%.1f steps/second' % ((steps_end-steps_start)/(time_end-time_start)) )
            print('%d steps in total' % steps_end)
            print('%.2f steps wasted' % wasted_percenteage)
            print('Current learning rate is %.5f' % model.lr.value())
            print('-------------------------------------------')
            print('Mini Model accuracy is %.3f' % last_mm_acc)
            saver.save(sess, path + '/models/' + run_name + '/model-' + str(init_num_upd + nupd) + '.cptk')
            average_training_steps = []

                  
        if write_sum:
            if nupd % 5 == 0 and nupd > 0:
                loss_history = np.array(loss_history)
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', 
                                  simple_value = float(np.mean(rewards_history)))
                summary.value.add(tag='Perf/ClippedReward', 
                                  simple_value = float(np.mean(clipped_rewards_history)))
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
                summary.value.add(tag='Losses/MM Loss', 
                                  simple_value = float(np.mean(mm_loss_hist)))
                summary_writer.add_summary(summary, nupd + init_num_upd)
                summary_writer.flush()
                
                loss_history, rewards_history, clipped_rewards_history, lengths_history = [], [], [], []
                training_steps_history = []
                mm_loss_hist = []
    return game, model, sess