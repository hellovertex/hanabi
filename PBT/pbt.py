import numpy as np
import tensorflow as tf
import random
import os
from collections import defaultdict
import pickle
from PPOAgent.Model import Model
def randomize_dict(d, min_val = 0.5, max_val = 2):
    for key in d:
        d[key] *= np.random.uniform(min_val, max_val)
    return d
def mutate_dict(d, mutations):
    d_new = dict(d)
    for key in d_new:
        mutation = random.choice(mutations)
        d_new[key] *= mutation
    return d_new
def sliding_average(data, n = 10):
    data_averaged = [np.mean(data[i : i + n]) for i in range(0, len(data) + 1 - n, n)]
    residue = len(data) % n
    data_averaged.append(np.mean(data[-residue:]))
    return np.array(data_averaged)

def sliding_average_to_dict(d, n = 10):
    new_d = {}
    for key in d:
        vals = d[key]
        vals_avg = sliding_average(vals, n)
        new_d[key] = vals_avg
        L = len(vals_avg)
    return new_d, L

def create_pool(nactions, nobs, nplayers, sess, MODEL_CONFIG_BASE, nmodels = 5, lr_range = [2e-3, 1e-4]):
    model_pool = []
    for i in range(nmodels):
        MODEL_CONFIG = dict(MODEL_CONFIG_BASE)
        MODEL_CONFIG['scope'] = 'agent%d' %i
        MODEL_CONFIG['lr'] = np.random.uniform(lr_range[0], lr_range[1])
        model_pool.append(Model(nactions, nobs, nplayers, sess = sess,
                         **MODEL_CONFIG))
    return model_pool

def train_pool(game, model_pool, summary_writer_pool, reward_weights_pool, k_pool, updates_until_comparasion,
               population_name = 'pbt_check', folder = './experiments/PBT/', save_every = 10):
    model_losses = [[] for _ in range(len(model_pool))]
    pool_results = [defaultdict(list) for _ in range(len(model_pool))]
    for model_num, model in enumerate(model_pool):
        rewards_config = reward_weights_pool[model_num]
        k = k_pool[model_num]
        game.reset_model(model, rewards_config)
        for nupd in range(1, updates_until_comparasion + 1):
            scores, rewards, lengths, custom_rewards = game.play_untill_train()
            state_probs, policy_loss, value_loss, policy_entropy, i = game.train_model(k, 1)
            losses = (policy_loss, value_loss, policy_entropy)
            pool_results[model_num]['scores'].append(np.mean(scores))
            pool_results[model_num]['rewards'].append(np.mean(rewards))
            pool_results[model_num]['lengths'].append(np.mean(lengths))
            
            for key in custom_rewards:
                pool_results[model_num][key].append(np.mean(custom_rewards[key]))
            if nupd % save_every == 0 and nupd > 0:
                summary_writer = summary_writer_pool[model_num]
                write_model_summary(summary_writer, model, pool_results[model_num], losses, 
                                    population_name, folder = folder, save_every = save_every)
    return np.array(model_losses), pool_results


def get_worst_and_best_model_nums(pool_results, nmodels, sort_by = 'scores', 
                                  evolve_ratio = 0.3, take_last = 10):
    sort_value = [np.mean(pool_results[model_num][sort_by][-take_last:]) for model_num in range(nmodels)]
    nmodels_to_evolve = int(evolve_ratio * nmodels)
    sorted_tuples = sorted(list(zip(sort_value, list(range(nmodels)))))
    sorted_model_nums = [tpl[1] for tpl in sorted_tuples]
    worst_model_nums = sorted_model_nums[:nmodels_to_evolve]
    best_model_nums = sorted_model_nums[-nmodels_to_evolve:]
    return sorted_model_nums
        
        
def draw_copy_pairs(worst_model_nums, best_model_nums):
    pairs = []
    for bad_model_num in worst_model_nums:
        good_model_num = random.choice(best_model_nums)
        pairs.append((bad_model_num, good_model_num))
    return pairs

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def update_k(pair, k_pool):
    new_k = k_pool[pair[1]]
    old_k = k_pool[pair[0]]
    k_pool[pair[0]] = new_k
    print('K: %d ---> %d' % (old_k, new_k))
    
def update_model_lr(pair, model_pool):
    bad_model = model_pool[pair[0]]
    good_model = model_pool[pair[1]]
    new_lr = good_model.lr.value()
    old_lr = bad_model.lr.value()
    bad_model.change_lr(new_lr)
    print('LR: %.5f ---> %.5f' % (old_lr, new_lr))

def update_rewards(pair, reward_weights_pool):
    good_reward_weights = reward_weights_pool[pair[1]]
    reward_weights_pool[pair[0]] = dict(good_reward_weights)
    
    
def mutate_models(model_pool, reward_weights_pool, k_pool, mutations):
    for model_num, model in enumerate(model_pool):
        model.change_lr(model.lr.value() * random.choice(mutations))
        k_pool[model_num] = max(1, int(k_pool[model_num] * random.choice(mutations)))
        reward_weights_pool[model_num] = mutate_dict(reward_weights_pool[model_num], mutations)
    return model_pool, reward_weights_pool, k_pool

def print_pool_results(pool_results, sorted_model_nums, title, time_taken):
    print('--------------------%s--------------------'%title)
    print('TIME TAKEN FOR TRAINING: %.2f' % time_taken)
    for model_num in sorted_model_nums:
        print('MODEL %d results:' % model_num)
        print('SCORES:%.2f'% np.mean(pool_results[model_num]['scores']), 
              ' LENGTHS:%.2f'% np.mean(pool_results[model_num]['lengths']),
              ' PLAY REWARD:%.2f'% np.mean(pool_results[model_num]['play_reward']), 
              ' HINT_REWARD:%.2f'% np.mean(pool_results[model_num]['hint_reward']), 
              ' DISCARD REWARD:%.2f'% np.mean(pool_results[model_num]['discard_reward']))
    print('--------------------%s--------------------'% ('-' * len(title)))
    
def write_model_summary(summary_writer, model, model_results, model_losses, population_name, 
                  folder = './experiments/PBT', save_every = 10):
    summary = tf.Summary()
    model_n_updates = model.sess.run(model.updates)
    for key in model_results:
        avg_val = np.mean(model_results[key][- save_every :])
        summary.value.add(tag = 'Perf/' + key,  simple_value = float(avg_val))
    summary.value.add(tag='Losses/Policy Loss', 
                      simple_value = float(model_losses[0]))
    summary.value.add(tag='Losses/Value Loss',
                      simple_value = float(model_losses[1]))
    summary.value.add(tag='Losses/Entropy', 
                      simple_value = float(model_losses[2]))
    summary_writer.add_summary(summary, model_n_updates)
    summary_writer.flush()
    
def write_pool_summary(summary_writer, model_pool, reward_weights_pool, k_pool, evolution_epoch,
                       population_name, folder = './experiments/PBT/'):
    summary = tf.Summary()
    reward_weight_keys = reward_weights_pool[0].keys()
    mean_lr = np.mean([model.lr.value() for model in model_pool])
    mean_k = np.mean(k_pool)
    summary.value.add(tag='Params/avg. lr', simple_value = float(mean_lr))
    summary.value.add(tag='Params/avg. k', simple_value = float(mean_k))
    for key in reward_weight_keys:
        vals = [reward_weights[key] for reward_weights in reward_weights_pool]
        mean_val = np.mean(vals)
        summary.value.add(tag='Params/avg. %s weight' % key, 
                          simple_value = float(mean_val))
    summary_writer.add_summary(summary, evolution_epoch)
    summary_writer.flush()
    
def save_population(model_pool, saver_pool, reward_weights_pool, k_pool, 
                    population_name, folder = './experiments/PBT/'):
    path = folder + population_name + '/'
    for model, saver, reward_dict, k in zip(model_pool, saver_pool, reward_weights_pool, k_pool):
        if not os.path.isdir(path + model.scope + '/'):
            os.makedirs(path + model.scope + '/')
        saver.save(sess, path + model.scope + '/model.cptk')
        with open(path + model.scope + '/rewards_dict.pkl', 'wb') as f:
            pickle.dump(reward_dict, f)
        with open(path + model.scope + '/k.pkl', 'wb') as f:
            pickle.dump(k, f)
            
def load_population(model_pool, saver_pool, population_name, folder = './experiments/PBT/'):
    
    reward_weights_pool = []
    k_pool = []
    path = folder + population_name + '/'
   
    for i, model in enumerate(model_pool):
        ckpt = tf.train.get_checkpoint_state(path + model.scope + '/')
        saver = saver_pool[i]
        saver.restore(model.sess, ckpt.model_checkpoint_path)
        with open(path + model.scope + '/rewards_dict.pkl', 'rb') as f:
            reward_weights_pool.append(pickle.load(f))
        with open(path + model.scope + '/k.pkl', 'rb') as f:
            k_pool.append(pickle.load(f))
            
    path_to_log = folder + population_name + '/params/'
    log_file_name = os.listdir(path_to_log)[0]
    start_epoch = 0
    log_file = tf.train.summary_iterator(path_to_log + log_file_name)
    for l in log_file:
        start_epoch = max(start_epoch, l.step)
    return reward_weights_pool, k_pool, start_epoch
    
