from collections import deque, defaultdict
import numpy as np
from tf_agents_lib.pyhanabi_env_wrapper import *
from tf_agents_lib import parallel_py_environment
from .gamesettings import *

def compute_game_score(obs_vec):
    score = 0
    fireworks = obs_vec[FIREWORKS_START : FIREWORKS_END]
    for c in range(NUM_COLORS):
        fireworks_color = np.array(fireworks[c * NUM_RANKS : (c + 1) * NUM_RANKS])
        pos = np.where(fireworks_color == 1)[0]
        if len(pos) != 0:
            score += pos[0] + 1
    return score

def mix_arrays(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    fin = []
    for i in range(arr1.shape[1]):
        fin.append(arr1[:, i])
        fin.append(arr2[:, i])
    fin = np.array(fin).swapaxes(0, 1)
    return fin
    
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

    
    
class SPGame():
    def __init__(self, model, load_env, rewards_config ={}):
        
        # create env
        self.env = parallel_py_environment.ParallelPyEnvironment([load_env] * model.nenvs)
        self.nenvs = model.nenvs
        self.model = model
        self.num_actions = model.num_actions
        # reset env and initialize observations
        self.ep_num = 0
        self.reset(rewards_config)
        self.reset_history()
        self.total_steps = 0
    
    def gen_noise_for_stepping(self, noisescale = 1):
        noise_ph_list = self.model.model.noise_list
        noise_val_list = []
        for noise_ph in noise_ph_list:
            noise_val_list.append((np.random.normal(0, noisescale, noise_ph.shape)))
        self.noise_val_list = noise_val_list

        
    def reset(self, rewards_config = {}):
        self.obs, _, self.legal_moves, self.ep_done, self.scores, self.ep_custom_rewards =\
        parse_timestep(self.env.reset(rewards_config ))
        self.last_episodes_custom_rewards = {k : [] for k in self.ep_custom_rewards}
        self.ep_stats = np.zeros((2, self.nenvs))
        self.ready_envs = set()
        self.last_episodes_reward = []
        self.last_episodes_length = []
        self.last_episodes_score = []
        self.gen_noise_for_stepping()
    
    def reset_history(self,):
        (self.mb_obs, self.mb_actions, self.mb_probs, self.mb_neglogps, self.mb_values,
         self.mb_rewards, self.mb_dones, self.mb_legal_moves) = [], [], [], [], [], [], [], []
            
    def play_turn(self,):
        
        actions, probs, neglops, values = self.model.step(self.obs, self.legal_moves, self.noise_val_list)
        values = values[:, 0]

        ts = self.env.step(np.array(actions))
        obs, rewards, legal_moves, dones, scores, custom_rewards = parse_timestep(ts)

        self.mb_obs.append(self.obs)
        self.mb_legal_moves.append(self.legal_moves)
        self.mb_actions.append(actions)
        self.mb_probs.append(probs)
        self.mb_neglogps.append(neglops)
        self.mb_values.append(values)
        self.mb_rewards.append(rewards)
        self.mb_dones.append(dones)
        
        
        for k in self.ep_custom_rewards:
            self.ep_custom_rewards[k] =  self.ep_custom_rewards[k] + custom_rewards[k]
       
        self.scores = scores
        self.total_steps += self.nenvs
        self.ep_stats += np.array([rewards, [1] * self.nenvs])
        
        self.obs, self.legal_moves, self.ep_done = obs, legal_moves, dones

    def play_untill_train(self, nepisodes = 90, noisescale = 1):
        self.gen_noise_for_stepping(noisescale)
        episodes_done = 0
        self.steps_to_train = 0
        self.last_episodes_reward = []
        self.last_episodes_score = []
        self.last_episodes_length = []
        self.last_episodes_custom_rewards = {k : [] for k in self.ep_custom_rewards}
        
        while episodes_done < nepisodes:
            self.play_turn()
            for j, d in enumerate(self.ep_done):
                if d:
                    self.finish_episode(j)
                    episodes_done += 1
        return (self.last_episodes_score, self.last_episodes_reward,
                self.last_episodes_length, self.last_episodes_custom_rewards)
    
    def finish_episode(self, j):
        # Updates last reward for env j. Moves data froms players' episode buffer to history buffer
        R, L = self.ep_stats[:, j]
        for k in self.ep_custom_rewards:
            self.last_episodes_custom_rewards[k].append(self.ep_custom_rewards[k][j])
            self.ep_custom_rewards[k][j] = 0
        self.last_episodes_reward.append(R)
        self.last_episodes_length.append(L)
        self.last_episodes_score.append(self.scores[j])
        self.ep_num += 1
        self.ep_done[j] = False
        self.ep_stats[:, j] = 0,0
        self.ep_stats[:, j] = 0,0
        
            
    def collect_data(self):
        
        mb_noise = np.array(self.noise_val_list)
        
        mb_actions = np.array(self.mb_actions)
        mb_actions = np.concatenate(mb_actions.T, 0)
        
        mb_obs = np.array(self.mb_obs)
        mb_obs = np.concatenate(mb_obs.swapaxes(0, 1), 0)
        
        mb_probs = np.array(self.mb_probs)
        mb_probs = np.concatenate(mb_probs.swapaxes(0, 1), 0)
        
        mb_neglogps = np.array(self.mb_neglogps)
        mb_neglogps = np.concatenate(mb_neglogps.swapaxes(0, 1), 0)
        
        
        mb_legal_moves = np.array(self.mb_legal_moves)
        mb_legal_moves = np.concatenate(mb_legal_moves.swapaxes(0, 1), 0)
        
        last_values = self.model.model.get_values(self.obs, mb_noise)[0]
        last_values = last_values[:, 0].reshape((-1))
        
        mb_values = np.array(self.mb_values).T
        
        mb_dones = np.array(self.mb_dones).T
        
        mb_rewards = np.array(self.mb_rewards).T
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        
        for t in reversed(range(mb_dones.shape[1])):
            nextnonterminal = 1.0 - mb_dones[:, t]
            if t == mb_dones.shape[1] - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[:, t + 1]
                
            delta = mb_rewards[:, t] + self.model.gamma * nextvalues * nextnonterminal - mb_values[:, t]
            mb_advs[:, t] = lastgaelam = delta + self.model.gamma * .95 * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values
        mb_dones = np.concatenate(mb_dones, 0)
        mb_rewards = np.concatenate(mb_returns, 0)
        mb_values = np.concatenate(mb_values, 0)
        dt = (mb_obs, mb_actions, mb_probs, mb_neglogps, mb_legal_moves, mb_values, mb_rewards,
               mb_dones, mb_noise)
        #for arr in dt:
        #    print(arr.shape)
        return (mb_obs, mb_actions, mb_probs, mb_neglogps, mb_legal_moves, mb_values, mb_rewards,
               mb_dones, mb_noise)
    

   
 
    