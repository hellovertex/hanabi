from collections import defaultdict
import numpy as np
from tf_agents_lib.pyhanabi_env_wrapper import *
from tf_agents_lib import parallel_py_environment


    
def parse_timestep(ts):
    # parse timestep, returns vectors for observation, rewards, legal_moves, dones
    dones = ts[0] == 2
    rewards = ts[1]
    legal_moves = ts[3]['mask']
    for i in range(len(legal_moves)):
        lm = legal_moves[i]
        lm = [-1e10 if lmi < 0 else 0 for lmi in lm]
        legal_moves[i] = lm
    obs = ts[3]['state']
    score = ts[3]['score']
    custom_rewards = ts[3]['custom_rewards']
    return obs, rewards, legal_moves, dones, score, custom_rewards


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

class Player():
    def __init__(self, num, nenvs):
        # params
        self.num = num
        self.nenvs = nenvs
        self.reset()
        
    def assign_model(self, model):
        self.model = model
        self.gamma = model.gamma
        self.reset()
        
    def reset(self):
        self.history_buffer =  [defaultdict(list) for _ in range(self.nenvs)]
        self.waiting = [False for _ in range(self.nenvs)]

    def generate_noise(self, noisescale = 1):
        noise_ph_list = self.model.model.noise_list
        noise_val_list = []
        for noise_ph in noise_ph_list:
            noise_val_list.append((np.random.normal(0, noisescale, noise_ph.shape)))
        self.noise_val_list = noise_val_list
        
    def step(self, legal_moves, obs, prev_dones, prev_rewards,) :
        # checks if player waits for reward from previous step, updates it if so
        # computes next actions, values, states and states_v, updates episode buffer
        for i, (w, p_r, p_d) in enumerate(zip(self.waiting, prev_rewards, prev_dones)):
            if w:
                self.history_buffer[i]['rewards'].append(p_r)
                self.history_buffer[i]['dones'].append(p_d)
       
        actions, probs, neglogps, values = self.model.step(obs, legal_moves, self.noise_val_list)
        values = values[:, 0]
        for i in range(self.nenvs):
            self.history_buffer[i]['obses'].append(obs[i])
            self.history_buffer[i]['actions'].append(actions[i])
            self.history_buffer[i]['probs'].append(probs[i])
            self.history_buffer[i]['neglogps'].append(neglogps[i])
            self.history_buffer[i]['values'].append(values[i])
            self.history_buffer[i]['masks'].append(prev_dones[i])
            self.history_buffer[i]['legal_moves'].append(legal_moves[i])
            
        self.waiting = [True for _ in range(self.nenvs)]
        
        return actions, probs, neglogps, values
        
    def get_training_data(self, ts_take):
            
        noise = np.array(self.noise_val_list)
        obses = np.concatenate([self.history_buffer[i]['obses'][ :ts_take] for i in range(self.nenvs)])
        actions = np.concatenate([self.history_buffer[i]['actions'][ :ts_take] for i in range(self.nenvs)])
        probs = np.concatenate([self.history_buffer[i]['probs'][ :ts_take] for i in range(self.nenvs)])
        neglogps = np.concatenate([self.history_buffer[i]['neglogps'][ :ts_take] for i in range(self.nenvs)])
        legal_moves = np.concatenate([self.history_buffer[i]['legal_moves'][ :ts_take] for i in range(self.nenvs)])
        
        dones = np.array([self.history_buffer[i]['dones'][ : ts_take] for i in range(self.nenvs)])
        rewards = np.array([self.history_buffer[i]['rewards'][ : ts_take] for i in range(self.nenvs)])
        values = np.array([self.history_buffer[i]['values'][ : ts_take] for i in range(self.nenvs)])
        last_values = np.array([self.history_buffer[i]['values'][ts_take] for i in range(self.nenvs)])
        
        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(dones.shape[1])):
            nextnonterminal = 1.0 - dones[:, t]
            if t == dones.shape[1] - 1:
                nextvalues = last_values
            else:
                nextvalues = values[:, t + 1]
            delta = rewards[:, t] + self.gamma * nextvalues * nextnonterminal - values[:, t]
            advs[:, t] = lastgaelam = delta + self.gamma * .95 * nextnonterminal * lastgaelam

        returns = advs + values
        dones = np.concatenate(dones, 0)
        returns = np.concatenate(returns, 0)
        values = np.concatenate(values, 0)
        self.reset()
        
        return (obses, actions, probs, neglogps, legal_moves, values, 
                returns, dones, noise)
    

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
        actions, probs, neglogps, values = player.step(self.legal_moves, self.obs, self.prev_dones[self.current_player], 
                                                       self.prev_rewards[self.current_player])
        #print(probs)
        self.steps_per_player[player.num] += 1
        ts = self.env.step(np.array(actions))
        obs, rewards, legal_moves, dones, scores, custom_rewards = parse_timestep(ts)

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
        for p in self.players:
            p.generate_noise(noisescale)
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
        for p in self.players:
            p.generate_noise(noisescale)
            
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
        (mb_obses, mb_actions, mb_probs, mb_neglogps, mb_legal_moves, mb_values, 
         mb_returns, mb_dones, mb_noise) = [], [], [], [], [], [], [], [], []
        
        ts_take = int(np.min(self.steps_per_player)) - 1
        
        #print('TS TAKE', ts_take)
        for p in self.players:
            if p.num not in player_nums:
                continue
                
            (obses, actions, probs, neglogps, legal_moves, values, 
                returns, dones, noise) = p.get_training_data(ts_take)
            mb_obses.append(obses)
            mb_actions.append(actions)
            mb_probs.append(probs)
            mb_neglogps.append(neglogps)
            mb_legal_moves.append(legal_moves)
            mb_values.append(values)
            mb_returns.append(returns)
            mb_dones.append(dones)
            mb_noise.append(noise)
 
        
        return (np.concatenate(mb_obses), np.concatenate(mb_actions), np.concatenate(mb_probs), 
                np.concatenate(mb_neglogps), np.concatenate(mb_legal_moves), np.concatenate(mb_values), 
                np.concatenate(mb_returns),np.concatenate(mb_dones), np.concatenate(mb_noise))
    

   
 
    