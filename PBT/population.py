from collections import deque, defaultdict
import numpy as np
import random
import os
from .util import *
from PPO.ppo import Model
from learning.learn_pool import train_model, write_into_buffer
import tensorflow as tf

class Population:

    def __init__(self, nactions, nobs, nplayers, sess, nmodels,
                 model_config_base, rewards_config_base,
                 random_attributes, folder = './experiments/PBT/', name = 'test_run/',
                 eval_by_last = 20):
        if name[-1] != '/':
            name = name + '/'
        self.name = name
        self.path = folder + name
        self.nmodels = nmodels
        self.random_attributes = random_attributes
        self.evolve_epochs = 0
        self.models = []
        for i in range(nmodels):
            config = dict(model_config_base)
            rewards_config = dict(randomize_dict(rewards_config_base, 0.66, 1.5))
            for attr in random_attributes:
                config[attr] = random_attributes[attr]()
                
            config['scope'] = 'agent%d' % i
            config['path'] = self.path
            config['rewards_config'] = rewards_config
            model = Model(nactions, nobs, nplayers, sess = sess, **config)
            self.models.append(model)
            
        self.avg_writer = tf.summary.FileWriter(self.path + '/avg params/')
        self.recent_results = [deque(maxlen = eval_by_last) for _ in range(nmodels)]
        self.history_buffer = [defaultdict(list) for _ in range(nmodels)]

        sess.run(tf.global_variables_initializer())
        for model in self.models:
            model.load_model()
        if len(os.listdir(self.path + 'avg params/')):
            log_file = tf.train.summary_iterator('./experiments/openhands/pbt_test/avg params/' +
                                                 os.listdir(self.path + 'avg params/')[0])
            for l in log_file:
                self.evolve_epochs = max(self.evolve_epochs, l.step)
             
        print('Created population. Initial epoch is %d' % self.evolve_epochs)
            
    def run_training(self, game, timesteps, episodes_to_train = 90, nsteps_to_train = None,
                     save_every = 2000, summary_every = 1000):
        nsteps_to_train_def = int(nsteps_to_train)
        for i in range(self.nmodels):
            model = self.models[i]
            for player in game.players:
                player.assign_model(model)
            game.reset(model.rewards_config)
            model_nsteps = model.sess.run(model.nsteps)
            if model_nsteps > 0:
                nsteps_to_train = int(model_nsteps)
            else:
                nsteps_to_train = nsteps_to_train_def
            save_every_upd = save_every // (nsteps_to_train * game.nenvs)
            summary_every_upd = summary_every // (nsteps_to_train * game.nenvs)
            start_ts = int(game.total_steps)
            nupd = 0
            while game.total_steps - start_ts <= timesteps:
                #print('will run for %d steps' % nsteps_to_train)
                training_stats = game.play_untill_train(episodes_to_train, nsteps_to_train)

                policy_loss, value_loss, policy_entropy = train_model(game, model, player_nums = 'all')
                write_into_buffer(self.history_buffer[i], training_stats, policy_loss, 
                                  value_loss, policy_entropy)
                self.recent_results[i].append(np.mean(training_stats['scores']))
                nupd += 1
                if (nupd %  save_every_upd) == 0:
                    print('Saving', model.scope)
                    model.save_model()
                if (nupd % summary_every_upd) == 0:
                    #print('writing summary for ', model.scope)
                    summary = tf.Summary()
                    for key in self.history_buffer[i]:
                        summary.value.add(tag = key,simple_value = np.nanmean(self.history_buffer[i][key]))
                    self.history_buffer[i] = defaultdict(list)
                    model_steps = model.sess.run(model.timesteps)
                    model_epochs = model.sess.run(model.train_epochs)
                    #print('model an for %d steps and %d train epochs' % (model_steps, model_epochs))
                    summary.value.add(tag = 'Perf/Updates', simple_value = model_epochs)
                    model.writer.add_summary(summary, model_steps)
                    model.writer.flush()
                    
     
    def rank_models(self, n_to_evolve = 4):
        model_results = [np.mean(result) for result in self.recent_results]
        sorted_indexes = np.argsort(model_results)
        pairs = []
        for bad_index in sorted_indexes[ : n_to_evolve]:
            pair = (bad_index, random.choice(sorted_indexes[-n_to_evolve : ]))
            pairs.append(pair)
            
        self.pairs = pairs
        self.inds_to_evolve = sorted_indexes[ : -n_to_evolve]
    
    def exploit(self, n_to_evolve = 4):
        for bad_ind, good_ind in self.pairs:
            bad_model = self.models[bad_ind]
            good_model = self.models[good_ind]
            print('%s exploits %s' % (bad_model.scope, good_model.scope))
            for attr in self.random_attributes:
                bad_attr_val = getattr(bad_model, attr)
                good_attr_val = getattr(good_model, attr)
                old_val = bad_model.sess.run(bad_attr_val)  # remove
                bad_model.sess.run(bad_attr_val.assign(good_attr_val))
                bad_model.sess.run(update_target_graph(good_model.scope, bad_model.scope))
                #print('%s was %f, became %f' % (attr, old_val, bad_model.sess.run(bad_attr_val)))
            
            #print(self.rewards_weights)
            self.models[bad_ind].change_rewards_config(dict(self.models[good_ind].rewards_config))
            #print(self.rewards_weights)
            
    def explore(self, mutation_fun = lambda: np.random.uniform(0.75, 1.25), mutation_prob = 0.075):
        for ind in self.inds_to_evolve:
            model = self.models[ind]
            for attr in self.random_attributes:
                if attr == 'k' or attr == 'nsteps':
                    if np.random.uniform() < 0.05:
                        mutation = self.random_attributes[attr]()
                    else:
                        mutation = 1
                else:
                    if np.random.uniform() < mutation_prob:
                        mutation = mutation_fun()
                    else:
                        mutation = 1
                attr_val = getattr(model, attr)
                old_val = model.sess.run(attr_val)  # remove
                model.sess.run(attr_val.assign(attr_val * mutation))
                #print('%s was %f, became %f' % (attr, old_val, model.sess.run(attr_val)))
                
            
            self.models[ind].change_rewards_config(mutate_dict(self.models[ind].rewards_config,
                                                                   mutation_fun, mutation_prob))
    def save_population(self,):
        avg_rewards_config = defaultdict(list)
        avg_params = defaultdict(list)
        for model in self.models:
            model.save_model()
            model.save_params_summary()
            model.save_rewards_config_summary()
            for key, val in model.rewards_config.items():
                avg_rewards_config[key].append(val)
            for key in self.random_attributes:
                val = model.sess.run(getattr(model, key))
                avg_params[key].append(val)
        avg_rewards_config = {k : np.mean(v) for k, v in avg_rewards_config.items()}
        avg_params = {k : np.mean(v) for k, v in avg_params.items()}
        avg_summary = tf.Summary()
        for key, val in avg_rewards_config.items():
            avg_summary.value.add(tag = 'Avg. rewards/' + key, simple_value = val)
        for key, val in avg_params.items():
            avg_summary.value.add(tag = 'Avg. params/' + key, simple_value = val)
            
        self.avg_writer.add_summary(avg_summary, self.evolve_epochs)
        self.avg_writer.flush()
        
        
    def run_epoch(self, game, nupdates, n_to_evolve, 
                  episodes_to_train = 32, nsteps_to_train = None,
                  save_every = 10, summary_every = 10):
        
        self.run_training(game, nupdates, episodes_to_train, nsteps_to_train,
                                save_every, summary_every)
        self.rank_models(n_to_evolve)
        self.exploit(n_to_evolve)
        self.explore()
        self.save_population()
        self.evolve_epochs += 1
        
    
    