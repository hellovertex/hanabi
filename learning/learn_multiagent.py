import itertools
import os
import random
import time
from collections import defaultdict
import pandas as pd
import numpy as np

from PPO.lstm_ppo import Model
from game.lstm_game import Game
import tensorflow as tf

from .util import *




def run_evaluation(models, game, nepisodes = 10):
    # support only 2 players now
    result_matrix = -np.ones((len(models), len(models)))
    model_nums = list(range(len(models)))
    for p1_model_num in model_nums:
        game.players[0].assign_model(models[p1_model_num])
        for p2_model_num in model_nums:
            game.players[1].assign_model(models[p1_model_num])
            
            game.reset()
            result_scores = np.mean(game.eval_results( episodes_per_env = nepisodes)['scores'])
            result_matrix[p1_model_num, p2_model_num] = result_scores


    return result_matrix


def train_one_epoch(game, models, history_buffers, updates_per_model = 100, summary_every = 10):
    model_nums = list(range(len(models)))
    for main_model_num, model_buffer in zip(model_nums, history_buffers):
        main_model = models[main_model_num]
        other_models = [m for m in models if m.scope != main_model.scope]
        main_player = random.choice(game.players)
        other_players = [p for p in game.players if p.num != main_player.num]
        main_player.assign_model(main_model)
        nsteps = main_model.nsteps
        main_player.reset()
        for other_player in other_players:
            other_player.assign_model(random.choice(other_models))
        print('Player%d plays with %s \nPlayer%d plays with %s' %(main_player.num, main_model.scope,
                                                                  other_player.num, other_player.model.scope))
        for upd in range(1, updates_per_model):
            training_stats = game.play_untill_train(nsteps = nsteps) # collect
            policy_loss, value_loss, policy_entropy = train_model(game, main_model, [main_player.num]) # train 
            write_into_buffer(model_buffer, training_stats, policy_loss, value_loss, policy_entropy) # store 
            if upd % summary_every == 0:
                summary = tf.Summary()
                for key in model_buffer:
                    summary.value.add(tag = key,simple_value = np.nanmean(model_buffer[key]))
                model_steps = main_model.sess.run(main_model.timesteps)
                summary.value.add(tag = 'Perf/Timesteps', simple_value = model_steps)
                model_epochs = main_model.sess.run(main_model.train_epochs)
                main_model.writer.add_summary(summary, model_epochs)
                main_model.writer.flush()
                history_buffers[main_model_num] = defaultdict(list)

def run_experiment(run_name, models_configs, env_config,  rewards_configs,
                   nupdates = 10000, updates_per_model = 500, reset_lr = False,
                   save_every = 1, summary_every = 20, evaluation_every = 5, eval_eps = 10,
                   folder = './experiments/multiagent/'):
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

    models = [Model(nactions, nobs, 1, sess = sess, **mc) for mc in models_configs]
    # create game
    game  = Game(nplayers, nenvs, load_env_fn, wait_rewards = True)
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
        train_one_epoch(game, models, history_buffers, updates_per_model, summary_every,)
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

            
        if nupd % evaluation_every == 0:
            matrix = run_evaluation(models, game, eval_eps)
            eval_result = pd.DataFrame(matrix, columns = [m.scope for m in models])
            eval_result.index = [m.scope for m in models]
            print('---------------%d---------------' % nupd)
            print('Matrix of models performance with each others:')
            print(eval_result)
            print('------------------------------' + '-' * len(str(nupd)))

