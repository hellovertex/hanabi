import numpy as np
from collections import defaultdict
from tf_agents_lib import parallel_py_environment
from .utils import parse_timestep, discount_with_dones
from .lstm_player import Player

class Game():
    def __init__(self, nplayers, nenvs, load_env, wait_rewards = True):
        # create env
        self.env = parallel_py_environment.ParallelPyEnvironment([load_env] * nenvs)
        # save params
        self.nenvs = nenvs
        self.nplayers = nplayers
        self.wait_rewards = wait_rewards
        self.players = [Player(i, nenvs) for i in range(self.nplayers)]
        self.total_steps = 0
        self.reset()
        
        
    def reset(self, rewards_config = {},):
        
        self.obs, _, self.legal_moves, self.ep_done, self.scores, self.ep_custom_rewards =\
        parse_timestep(self.env.reset(rewards_config))
        self.current_player = 0
        self.steps_per_player = np.zeros(self.nplayers)
        self.prev_rewards = np.zeros((self.nplayers, self.nenvs))
        self.prev_dones = np.ones((self.nplayers, self.nenvs), dtype =  'bool')
        self.ep_stats = np.zeros((2, self.nenvs))
        self.ready_envs = set()
        self.last_episodes_stats = defaultdict(list)
        for p in self.players:
            p.reset()

            
           
    def play_turn(self,):
        # current player plays one turn
        player = self.players[self.current_player]
        #print('player', player.num)
        actions, probs, alogps, values = player.step(self.legal_moves, self.obs, self.prev_dones[self.current_player], 
                                                       self.prev_rewards[self.current_player])

        
        self.steps_per_player[player.num] += 1
        ts = self.env.step(np.array(actions))
        obs, rewards, legal_moves, dones, scores, custom_rewards = parse_timestep(ts)
        #print('R', rewards)
        #print('D', dones)
        for k in self.ep_custom_rewards:
            self.ep_custom_rewards[k] =  self.ep_custom_rewards[k] + custom_rewards[k]

        self.scores = scores

        self.total_steps += self.nenvs
        if self.wait_rewards:
            self.prev_rewards[self.current_player] = 0
            self.prev_rewards += rewards
        else:
            self.prev_rewards[self.current_player] = rewards
        self.ep_stats += np.array([rewards, [1] * self.nenvs])

        self.prev_dones[self.current_player] = 0
        self.prev_dones += dones
        
        self.current_player = (self.current_player + 1) % self.nplayers
        self.obs, self.legal_moves, self.ep_done = obs, legal_moves, dones

    def eval_results(self, episodes_per_env = 6, noisescale = 1):
        # runs the game untll gall envs collect enough data for training
        self.reset()
        self.players[0].generate_noise(noisescale = noisescale)
        self.players[1].generate_noise(self.players[0].noise_val_list, noisescale)
        episodes_done = np.zeros(self.nenvs)
        record_env = np.ones(self.nenvs, dtype = 'bool')
        self.last_episodes_stats = defaultdict(list)
        
        while min(episodes_done) < episodes_per_env:
            self.play_turn()
            for j, d in enumerate(self.ep_done):
                if d:
                    if episodes_done[j] >= episodes_per_env:
                        record_env[j] = False
                    self.finish_episode(j, record_env[j])
                    episodes_done[j] += 1

        return self.last_episodes_stats
    
    def play_untill_train(self, nepisodes = 90, nsteps = None, noisescale = 1):
        # runs the game untll gall envs collect enough data for training
        #self.reset()
        
        self.last_episodes_stats = defaultdict(list)
        self.players[0].generate_noise(noisescale = noisescale)
        self.players[1].generate_noise(self.players[0].noise_val_list, noisescale)
            
        self.first_player = self.current_player
        self.steps_per_player = np.zeros(self.nplayers)
        
        episodes_done = 0
        steps_done = np.zeros(self.nplayers)
        ready = False
        
        while not ready:
            steps_done[self.current_player] += 1
            self.play_turn()
            if nsteps is not None:
                ready = min(steps_done > nsteps)
            for j, d in enumerate(self.ep_done):
                if d:
                    self.finish_episode(j, True)
                    episodes_done += 1
                    if nsteps is None:
                        ready = episodes_done >= nepisodes
                        
        return self.last_episodes_stats
    def finish_episode(self, j, record = True):
        # Updates last reward for env j. Moves data froms players' episode buffer to history buffer
        R, L = self.ep_stats[:, j]
        if record:
            for k in self.ep_custom_rewards:
                self.last_episodes_stats[k].append(self.ep_custom_rewards[k][j])
                self.ep_custom_rewards[k][j] = 0
            self.last_episodes_stats['rewards'].append(R)
            self.last_episodes_stats['lengths'].append(L)
            self.last_episodes_stats['scores'].append(self.scores[j])
        
        self.ep_done[j] = False
        self.ep_stats[:, j] = 0,0
        
        for p in self.players:
            if p.waiting[j]:
                p.history_buffer[j]['rewards'].append(self.prev_rewards[p.num][j])
                p.history_buffer[j]['dones'].append(self.prev_dones[p.num][j])
                
            p.waiting[j] = False
            
    
    def collect_data(self, player_nums = 'all'):
        if player_nums == 'all':
            player_nums = list(range(self.nplayers))
        (mb_obses, mb_actions, mb_probs, mb_alogps, mb_legal_moves, mb_values, 
         mb_returns, mb_dones, mb_masks, mb_states, mb_states_v, mb_noise) = [], [], [], [], [], [], [], [], [], [], [], []
        
        ts_take = int(np.min(self.steps_per_player)) - 1
        
        #print('TS TAKE', ts_take)
        for p in self.players:
            if p.num not in player_nums:
                continue
                
            (obses, actions, probs, alogps, legal_moves, values, 
                returns, dones, masks, states, states_v, noise) = p.get_training_data(ts_take)
            mb_obses.append(obses)
            mb_actions.append(actions)
            mb_probs.append(probs)
            mb_alogps.append(alogps)
            mb_legal_moves.append(legal_moves)
            mb_values.append(values)
            mb_returns.append(returns)
            mb_dones.append(dones)
            mb_masks.append(masks)
            mb_states.append(states)
            mb_states_v.append(states_v)
            mb_noise.append(noise)
        if states[0] is not None:
            #print('concating states')
            mb_states = np.concatenate(mb_states, 1)
            if states_v[0] is not None:
                mb_states_v = np.concatenate(mb_states_v, 1)
        #print('Game output states shape', np.shape(states), np.shape(states_v))
        return (np.concatenate(mb_obses), np.concatenate(mb_actions), np.concatenate(mb_probs), 
                np.concatenate(mb_alogps), np.concatenate(mb_legal_moves), np.concatenate(mb_values), 
                np.concatenate(mb_returns),np.concatenate(mb_dones), np.concatenate(mb_masks),
                np.array(mb_states), np.array(mb_states_v), np.concatenate(mb_noise))
