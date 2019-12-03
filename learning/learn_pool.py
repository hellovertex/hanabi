import itertools
import os
import random
import time
from collections import defaultdict
import pandas as pd
import numpy as np

from PPO.ppo import Model as Model
from PPO.lstm_ppo import Model as ModelLSTM
from game.game import Game as Game
from game.lstm_game import Game as GameLSTM

import tensorflow as tf

from .util import *




def run_evaluation(models, game, nepisodes = 10):
    # support only 2 players now
    result_matrix = -np.ones((len(models), len(models)))
    model_nums = list(range(len(models)))
    for p1_model_num in model_nums:
        
        game.players[0].assign_model(models[p1_model_num])
        if game.nplayers == 2:
            for p2_model_num in model_nums:
                game.players[1].assign_model(models[p1_model_num])
                game.reset()
                result_scores = np.mean(game.eval_results( episodes_per_env = nepisodes)['scores'])
                result_matrix[p1_model_num, p2_model_num] = result_scores
        else:
            game.reset()
            result_scores = np.mean(game.eval_results( episodes_per_env = nepisodes)['scores'])
            result_matrix[p1_model_num, p1_model_num] = result_scores

    return result_matrix


def train_one_epoch(game, models, history_buffers, updates_in_epoch = 100,
                    episodes = 90, nsteps = None,
                    method = 'self play', epochs_for_ma = 5,):
    # methods:
    #    single agent -- game will be played with single agent
    #    self play -- each model will play with its copy, learning 
    #    multi agent -- each model will play with all other models
    #    if train is True all games will be used for training

    if method == 'single agent':
        for model, model_buffer in zip(models, history_buffers):
            # for sigle agent game uses one model
            game.players[0].assign_model(model)
            if hasattr(model, 'nsteps'):
                nsteps = model.nsteps
            training_stats = game.play_untill_train(episodes, nsteps) # collect
            policy_loss, value_loss, policy_entropy = train_model(game, model) # train 
            write_into_buffer(model_buffer, training_stats, policy_loss, value_loss, policy_entropy) # store 
        
    elif method == 'self play':
        for model, model_buffer in zip(models, history_buffers):
            for player in game.players:
                player.assign_model(model)
            if hasattr(model, 'nsteps'):
                nsteps = model.nsteps
                
            training_stats = game.play_untill_train(episodes, nsteps) # collect
            policy_loss, value_loss, policy_entropy = train_model(game, model) # train 
            write_into_buffer(model_buffer, training_stats, policy_loss, value_loss, policy_entropy) # store 
            
    elif method == 'multi agent':
        model_nums = list(range(len(models)))
        # main_model_num is number of the model which will train by playing with
        # other models (including itself) are frozen.
        # it will play from position of main_player_num
        for main_model_num, model_buffer in zip(model_nums, history_buffers):
            main_model = models[main_model_num]
            main_player = random.choice(game.players)
            main_player.assign_model(main_model)
            if hasattr(main_model, 'nsteps'):
                nsteps = model.nsteps
            
            for _ in range(epochs_for_ma):
                for other_player in game.players:
                    if other_player.num != main_player.num:
                        other_player.assign(random.choice(models))
                train_players = [p.num for p in game.players if p.model.scope == main_model.scope]
                training_stats = game.play_untill_train(episodes, nsteps) # collect
                policy_loss, value_loss, policy_entropy = train_model(game, mode, train_players) # train 
                write_into_buffer(model_buffer, training_stats, policy_loss, value_loss, policy_entropy) # store 


def run_experiment(run_name, models_configs, env_config,  rewards_configs,
                   nupdates = 10000,  episodes = 90, nsteps = None, change_lr = False,
                   save_every = 100, summary_every = 10, evaluation_every = 200, eval_eps = 10,
                   folder = './experiments/openhands/', method = 'self play'):
    # set session
    tf.reset_default_graph()
    sess = tf.Session()
    # setting up environment depending variables
    load_env_fn = make_load_hanabi_fn(env_config)
    env = load_env_fn()
    nobs, nactions, nplayers = get_env_spec(env_config)
    nenvs = models_configs[0]['nenvs']
    path = folder + env_config['environment_name'] + '-' + str(env_config['num_players'])
    path += '/' + run_name + '/'
    # create model
    for config in models_configs:
        config['path'] = path
    modelfn = ModelLSTM if models_configs[0]['lstm_layers'] else Model
    gamefn = GameLSTM if models_configs[0]['lstm_layers'] else Game
    print(modelfn)
    models = [modelfn(nactions, nobs, nplayers, sess = sess, **mc) for mc in models_configs]
    # create game
    if method == 'single agent':
        game = gamefn(1, nenvs, load_env_fn)
    else:
        game  = gamefn(nplayers, nenvs, load_env_fn, wait_rewards = True)
    game.reset(rewards_configs)
    sess.run(tf.global_variables_initializer())
    # make savers, summaries, history buffers
    history_buffers = [defaultdict(list) for _ in range(len(models))]
    # try to load models
    for model in models:
        model.load_model()
    # run training
    steps_start, time_start = game.total_steps, time.time()
    for nupd in range(1, nupdates):
        # trains each model once
        train_one_epoch(game, models, history_buffers, episodes, nsteps, method )
        # save models once in a while
        if nupd % save_every == 0:
            speed = (game.total_steps - steps_start) / (time.time() - time_start)
            print()
            print('Savinig models after %d epochs of training' % nupd) 
            print('Average env speed is %d ts/second' % (speed))
            steps_start, time_start = game.total_steps, time.time()
            for model in models:
                model.save_model()
                model.save_params_summary()
        # save summaries, more often than models
        if nupd % summary_every == 0:
            summary = tf.Summary()
            for i in range(len(models)):
                model = models[i]
                for key in history_buffers[i]:
                    summary.value.add(tag = key,simple_value = np.nanmean(history_buffers[i][key]))
                model_steps = model.sess.run(model.timesteps)
                summary.value.add(tag = 'Perf/Timesteps', simple_value = model_steps)
                model_epochs = model.sess.run(model.train_epochs)
                model.writer.add_summary(summary, model_epochs)
                model.writer.flush()
                history_buffers = [defaultdict(list) for _ in range(len(models))]
            
        if nupd % evaluation_every == 0:
            matrix = run_evaluation(models, game, eval_eps)
            eval_result = pd.DataFrame(matrix, columns = [m.scope for m in models])
            eval_result.index = [m.scope for m in models]
            print('---------------%d---------------' % nupd)
            print('Matrix of models performance with each others:')
            print(eval_result)
            print('------------------------------' + '-' * len(str(nupd)))

