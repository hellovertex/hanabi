import os
import numpy as np
from collections import defaultdict
from tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper
from hanabi_learning_environment import rl_env
import tensorflow as tf
def check_create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def make_load_hanabi_fn(ENV_CONFIG):
    return lambda: PyhanabiEnvWrapper(rl_env.make(**ENV_CONFIG))

def get_env_spec(ENV_CONFIG,):
    env = PyhanabiEnvWrapper(rl_env.make(**ENV_CONFIG))
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    nactions = action_spec.maximum + 1 - action_spec.minimum
    nobs = obs_spec['state'].shape[0]
    nplayers = ENV_CONFIG['num_players']
    
    print('GAME PARAMETERS: \n  observation length = %d\n  number of actions = %d\n  number of players = %d' %
          (nobs, nactions, nplayers))
    return nobs, nactions, nplayers

def write_into_buffer(buffer, training_stats, policy_loss, value_loss, policy_entropy):
    # writes all data into buffer dict
    buffer['Perf/Score'].append(np.mean(training_stats['scores']))
    buffer['Perf/Reward'].append(np.mean(training_stats['rewards']))
    buffer['Perf/Length'].append(np.mean(training_stats['lengths']))
    buffer['Perf/Reward by "play"'].append(np.mean(training_stats['play_reward']))
    buffer['Perf/Reward by "discard"'].append(np.mean(training_stats['discard_reward']))
    buffer['Perf/Reward by "hint"'].append(np.mean(training_stats['hint_reward']))
    #buffer['Perf/Updates per batch'].append(k_trained)
    #buffer['Perf/Updates done'].append(updates)
    #buffer['Losses/KL loss'].append(np.mean(kl))
    buffer['Losses/Policy loss'].append(np.mean(policy_loss))
    buffer['Losses/Value loss'].append(np.mean(value_loss))
    buffer['Losses/Policy entropy'].append(np.mean(policy_entropy))           
            
def train_model(game, model,  player_nums = 'all'):
    if model.type == 'lstm':
        (mb_obs, mb_actions, mb_probs, mb_logp, mb_legal_moves, mb_values, mb_returns,
         mb_dones, mb_masks, mb_states, mb_states_v, mb_noise) = game.collect_data(player_nums)
        
        policy_loss, value_loss, policy_entropy = model.train(mb_obs, mb_actions, mb_probs, mb_logp, 
                                                              mb_legal_moves, mb_masks, mb_values, mb_returns,
                                                              mb_states, mb_states_v, mb_noise)
    else:
        (mb_obs, mb_actions, mb_probs, mb_logp, mb_legal_moves, mb_values, mb_returns,
         mb_dones, mb_noise) = game.collect_data(player_nums)
    
        policy_loss, value_loss, policy_entropy = model.train(mb_obs, mb_actions, mb_probs, mb_logp, 
                                                              mb_legal_moves, mb_values, mb_returns, mb_noise)

    
    return policy_loss, value_loss, policy_entropy
        
        