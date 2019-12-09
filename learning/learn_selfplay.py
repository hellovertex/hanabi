import itertools
import os
import random
import time
from collections import defaultdict
import numpy as np

from PPO.lstm_ppo import Model
from game.lstm_game import Game
import tensorflow as tf

from .util import *




def eval_model(model, game, nepisodes = 10):
    # support only 2 players now
    for p in game.players:
        p.assign_model(model)
    game.reset()
    result_scores = np.mean(game.eval_results( episodes_per_env = nepisodes)['scores'])
    return result_scores


def train_one_epoch(game, model, history_buffer):
    for player in game.players:
        player.assign_model(model)
    nsteps = model.nsteps
    training_stats = game.play_untill_train(nsteps = nsteps) # collect
    policy_loss, value_loss, policy_entropy = train_model(game, model) # train 
    write_into_buffer(history_buffer, training_stats, policy_loss, value_loss, policy_entropy) # store 

def run_experiment(run_name, model_config, env_config,  rewards_configs,
                   nupdates = 10000, reset_lr = False,
                   save_every = 100, summary_every = 10, evaluation_every = 200, eval_eps = 10,
                   folder = './experiments/openhands/'):
    # set session
    tf.reset_default_graph()
    sess = tf.Session()
    # setting up environment depending variables
    load_env_fn = make_load_hanabi_fn(env_config)
    env = load_env_fn()
    nobs, nactions, nplayers = get_env_spec(env_config)
    nenvs = model_config['nenvs']
    path = folder + env_config['environment_name'] + '-' + str(env_config['num_players'])
    path += '/' + run_name + '/'
    model_config['path'] = path
    # create model
    model = Model(nactions, nobs, nplayers, sess = sess, **model_config)
    history_buffer = defaultdict(list)
    game  = Game(nplayers, nenvs, load_env_fn, wait_rewards = True)
    game.reset(rewards_configs)
    sess.run(tf.global_variables_initializer())
    # make savers, summaries, history buffers
    history_buffer = defaultdict(list)
    # try to load models
    model.load_model()
    # run training
    steps_start, time_start = game.total_steps, time.time()
    for nupd in range(1, nupdates):
        # trains each model once
        train_one_epoch(game, model, history_buffer)
        # save models once in a while
        if nupd % save_every == 0:
            speed = (game.total_steps - steps_start) / (time.time() - time_start)
            print()
            print('Savinig model after %d epochs of training' % nupd) 
            print('Average env speed is %d ts/second' % (speed))
            steps_start, time_start = game.total_steps, time.time()
            model.save_model()
            model.save_params_summary()
        # save summaries, more often than models
        if nupd % summary_every == 0:
            summary = tf.Summary()
            for key in history_buffer:
                summary.value.add(tag = key,simple_value = np.nanmean(history_buffer[key]))
            model_steps = model.sess.run(model.timesteps)
            summary.value.add(tag = 'Perf/Timesteps', simple_value = model_steps)
            model_epochs = model.sess.run(model.train_epochs)
            model.writer.add_summary(summary, model_epochs)
            model.writer.flush()
            history_buffer = defaultdict(list)
            
        if nupd % evaluation_every == 0:
            result_scores = eval_model(model, game, eval_eps)
            print('---------------%d---------------' % nupd)
            print("Model's performance is:")
            print(result_scores)
            print('------------------------------' + '-' * len(str(nupd)))

