import numpy as np
from collections import defaultdict

class Player():
    def __init__(self, num, nenvs):
        print('normal playa' )
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
        noise_ph_list = self.model.network.noise_list
        noise_val_list = []
        for noise_ph in noise_ph_list:
            noise_val_list.append((np.random.normal(0, noisescale, noise_ph.shape)))
        self.noise_val_list = noise_val_list
        
    def step(self, legal_moves, obs, prev_dones, prev_rewards,) :
        # checks if player waits for reward from previous step, updates it if so
        # computes next actions, values, probs, updates episode buffer
        for i, (w, p_r, p_d) in enumerate(zip(self.waiting, prev_rewards, prev_dones)):
            if w:
                self.history_buffer[i]['rewards'].append(p_r)
                self.history_buffer[i]['dones'].append(p_d)
       
        actions, probs, alogps, values = self.model.step(obs, legal_moves, self.noise_val_list)
        values = values[:, 0]
        for i in range(self.nenvs):
            self.history_buffer[i]['obses'].append(obs[i])
            self.history_buffer[i]['actions'].append(actions[i])
            self.history_buffer[i]['probs'].append(probs[i])
            self.history_buffer[i]['alogps'].append(alogps[i])
            self.history_buffer[i]['values'].append(values[i])
            self.history_buffer[i]['masks'].append(prev_dones[i])
            self.history_buffer[i]['legal_moves'].append(legal_moves[i])
            
        self.waiting = [True for _ in range(self.nenvs)]
        
        return actions, probs, alogps, values
        
    def get_training_data(self, ts_take):
            
        noise = np.array(self.noise_val_list)
        obses = np.concatenate([self.history_buffer[i]['obses'][ :ts_take] for i in range(self.nenvs)])
        actions = np.concatenate([self.history_buffer[i]['actions'][ :ts_take] for i in range(self.nenvs)])
        probs = np.concatenate([self.history_buffer[i]['probs'][ :ts_take] for i in range(self.nenvs)])
        alogps = np.concatenate([self.history_buffer[i]['alogps'][ :ts_take] for i in range(self.nenvs)])
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
 
        return (obses, actions, probs, alogps, legal_moves, values, 
                returns, dones, noise)