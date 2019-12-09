import numpy as np
from collections import defaultdict

class Player():
    def __init__(self, num, nenvs):
        # params
        self.num = num
        self.nenvs = nenvs
        self.reset()
        
    def assign_model(self, model, belief_model = None):
        self.model = model
        self.gamma = model.gamma
        if belief_model is not None:
            self.belief_model = belief_model
            self.use_beliefs = True
        else:
            self.use_beliefs = False
        self.states = self.model.init_state
        self.states_v = self.model.init_state_v
        #print('states after assignment', self.states, self.states_v)
        self.reset()
        
    def reset(self):
        self.history_buffer =  [defaultdict(list) for _ in range(self.nenvs)]
        self.waiting = [False for _ in range(self.nenvs)]
        if hasattr(self, 'model'):
            self.states = self.model.init_state
            self.states_v = self.model.init_state_v

    def add_beliefs(self, obs, prev_actions, prev_obs, beliefs_prob_dict):
        print('adding beliefs')
        belief_model_output = self.belief_model.compute_action_probs(prev_obs)
        belief_model_output[prev_actions == -1] = 1 / 11
        belief_model_output = np.array([belief_model_output[ienv, :, :, prev_actions[ienv]] 
                                        for ienv in range(self.nenvs)])
        # print('sh', np.shape(belief_model_output))
        hand_probs = beliefs_prob_dict['hand_probs']
        first_prior = beliefs_prob_dict['first_prior']
        second_prior = beliefs_prob_dict['second_prior']
        first_cond = beliefs_prob_dict['first_cond']
        second_cond = beliefs_prob_dict['second_cond']
        
        action_prior_prob = np.sum(np.sum(belief_model_output * hand_probs, 1), 1).reshape((-1, 1))
        action_cond_first = np.sum(belief_model_output * second_cond[:, :], 2)
        action_cond_second = np.sum(belief_model_output * first_cond[:, :], 1)
        first_card_probs = first_prior[:, :, 0] * action_cond_first/action_prior_prob
        second_card_probs = second_prior[:, :, 0] * action_cond_second/action_prior_prob
        obs_ext = np.concatenate([first_card_probs, second_card_probs, obs], 1)
        return obs_ext
        
    def generate_noise(self, noise = None, noisescale = 1):
        if noise is None:
            noise_ph_list = self.model.step_network.noise_list
            noise_val_list = []
            for noise_ph in noise_ph_list:
                noise_val_list.append((np.random.normal(0, noisescale, noise_ph.shape)))
        else:
            noise_val_list = noise
        self.noise_val_list = noise_val_list
        #print(np.shape(self.noise_val_list[0]))
        #print('player %d noise' % self.num, self.noise_val_list)
    def step(self, legal_moves, obs, prev_dones, prev_rewards,
             prev_actions = None, prev_obs = None, beliefs_prob_dict = {}) :
        # checks if player waits for reward from previous step, updates it if so
        # computes next actions, values, probs, updates episode buffer
        for i, (w, p_r, p_d) in enumerate(zip(self.waiting, prev_rewards, prev_dones)):
            if w:
                self.history_buffer[i]['rewards'].append(p_r)
                self.history_buffer[i]['dones'].append(p_d)

        if self.use_beliefs:
            obs_ext = self.add_beliefs(obs, prev_actions, prev_obs, beliefs_prob_dict)
        else:
            obs_ext = obs
        actions, probs, alogps, values, states, states_v  = self.model.step(obs_ext, legal_moves, prev_dones,
                                                                            self.states, self.states_v,
                                                                            self.noise_val_list)
        values = values[:, 0]
        #print('A', actions)
        #print('S', self.states)
        for i in range(self.nenvs):
            self.history_buffer[i]['obses'].append(obs[i])
            self.history_buffer[i]['obses_ext'].append(obs_ext[i])
            self.history_buffer[i]['actions'].append(actions[i])
            self.history_buffer[i]['probs'].append(probs[i])
            self.history_buffer[i]['alogps'].append(alogps[i])
            self.history_buffer[i]['values'].append(values[i])
            self.history_buffer[i]['masks'].append(prev_dones[i])
            self.history_buffer[i]['legal_moves'].append(legal_moves[i])
            
            if len(states):
                #print('saving states')
                self.history_buffer[i]['states'].append(np.swapaxes(self.states, 0, 1)[i])
                if self.states_v is None:
                    self.history_buffer[i]['states_v'].append(None)
                else:
                    self.history_buffer[i]['states_v'].append(np.swapaxes(self.states_v, 0, 1)[i])
            else:
                self.history_buffer[i]['states'].append(None)
                self.history_buffer[i]['states_v'].append(None)
                
        if len(states):
            self.states = states
            self.states_v = states_v
        self.waiting = [True for _ in range(self.nenvs)]
        
        return actions, probs, alogps, values
        
    def get_training_data(self, ts_take):
            
        noise = np.array(self.noise_val_list)
        obses = np.concatenate([self.history_buffer[i]['obses'][ :ts_take] for i in range(self.nenvs)])
        obses_ext = np.concatenate([self.history_buffer[i]['obses_ext'][ :ts_take] for i in range(self.nenvs)])
        actions = np.concatenate([self.history_buffer[i]['actions'][ :ts_take] for i in range(self.nenvs)])
        masks = np.concatenate([self.history_buffer[i]['masks'][ :ts_take] for i in range(self.nenvs)])
        probs = np.concatenate([self.history_buffer[i]['probs'][ :ts_take] for i in range(self.nenvs)])
        alogps = np.concatenate([self.history_buffer[i]['alogps'][ :ts_take] for i in range(self.nenvs)])
        legal_moves = np.concatenate([self.history_buffer[i]['legal_moves'][ :ts_take] for i in range(self.nenvs)])
        
        states = [self.history_buffer[i]['states'][0] for i in range(self.nenvs)]
        states_v = [self.history_buffer[i]['states_v'][0] for i in range(self.nenvs)]
        if states[0] is not None:
            states = np.swapaxes(states, 0, 1)
            if states_v[0] is not None:
                states_v = np.swapaxes(states_v, 0, 1 )
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
        self.history_buffer =  [defaultdict(list) for _ in range(self.nenvs)]
        return (obses, obses_ext, actions, probs, alogps, legal_moves, values, 
                returns, dones, masks, states, states_v, noise)