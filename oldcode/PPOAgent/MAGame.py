from collections import deque
from run_experiment import format_legal_moves
import numpy as np
from PyHanabiWrapper import *
from tf_agents.environments import parallel_py_environment
from collections import defaultdict

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

class MAPlayer():
    def __init__(self, num, model):
        # params  schedule par or turn
        self.num = num
        self.model = model
        self.gamma = model.gamma
        self.nsteps = model.nsteps
        self.nenvs = model.nenvs
        self.use_lstm = False if model.init_state is None else True
        self.reset()
        
    def reset(self):
        # resets history, waiting status and states  
        self.history_buffer =  [defaultdict(list) for _ in range(self.nenvs)]
        self.waiting = [False for _ in range(self.nenvs)]
        self.states = self.model.init_state
        self.states_v = self.model.init_state_v
        if not self.use_lstm:
            self.states = None
            self.states_v = None
    
    def step(self, legal_moves, obs, prev_dones, prev_rewards, noise = None, temp = 1.):
        # checks if player waits for reward from previous step, updates it if so
        # computes next actions, values, states and states_v, updates episode buffer
        for i, (w, p_r, p_d) in enumerate(zip(self.waiting, prev_rewards, prev_dones)):
            if w:
                self.history_buffer[i]['rewards'].append(p_r)
                self.history_buffer[i]['dones'].append(p_d)
                
        actions, probs, neglogps, values, states, states_v = self.model.step(obs, self.states, prev_dones, 
                                                                      self.states_v, legal_moves, noise, temp)
        values = values[:, 0]
        for i in range(self.nenvs):
            self.history_buffer[i]['obses'].append(obs[i])
            self.history_buffer[i]['actions'].append(actions[i])
            self.history_buffer[i]['probs'].append(probs[i])
            self.history_buffer[i]['neglogps'].append(neglogps[i])
            self.history_buffer[i]['values'].append(values[i])
            self.history_buffer[i]['masks'].append(prev_dones[i])
            self.history_buffer[i]['legal_moves'].append(legal_moves[i])
            if self.use_lstm:
                self.history_buffer[i]['states'].append(np.swapaxes(self.states, 0, 1)[i])
                if self.states_v is None:
                    self.history_buffer[i]['states_v'].append(None)
                else:
                    self.history_buffer[i]['states_v'].append(np.swapaxes(self.states_v, 0, 1)[i])
            else:
                self.history_buffer[i]['states'].append(None)
                self.history_buffer[i]['states_v'].append(None)
        if self.use_lstm:
            self.states = states
            self.states_v = states_v       
            
        self.waiting = [True for _ in range(self.nenvs)]

        return actions
    
    def get_training_data(self):
        # collects training data, clears history
        obses, actions,probs,neglogps, values, rewards, dones, masks, states, states_v, noise, legal_moves = \
        [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(self.nenvs):
            obses.append(self.history_buffer[i]['obses'][ : self.model.nsteps])
            actions.append(self.history_buffer[i]['actions'][ : self.model.nsteps])
            probs.append(self.history_buffer[i]['probs'][ : self.model.nsteps])
            neglogps.append(self.history_buffer[i]['neglogps'][ : self.model.nsteps])
            values.append(self.history_buffer[i]['values'][ : self.model.nsteps + 1])
            rewards.append(self.history_buffer[i]['rewards'][ : self.model.nsteps])
            dones.append(self.history_buffer[i]['dones'][ : self.model.nsteps])
            masks.append(self.history_buffer[i]['masks'][ : self.model.nsteps])
            states.append(self.history_buffer[i]['states'][0])
            states_v.append(self.history_buffer[i]['states_v'][0])
            legal_moves.append(self.history_buffer[i]['legal_moves'][ : self.model.nsteps])
            #noise.append(self.history_buffer[i]['noise'][ : self.model.nsteps])
        if self.use_lstm:
            states = np.swapaxes(states, 0, 1)
            if states_v[0] is not None:
                states_v = np.swapaxes(states_v, 0, 1 )
        self.history_buffer = [defaultdict(list) for _ in range(self.nenvs)]
        return (obses, actions, probs, neglogps, values, rewards, 
                dones, masks, states, states_v, legal_moves)
    
    
class MAGame():
    def __init__(self, num_players, models, load_env, wait_rewards = True,
                 train_schedule = 'par', updates_wait = 10, train_players = None):
        
        self.env = parallel_py_environment.ParallelPyEnvironment([load_env] * models[0].nenvs)
        # save params
        self.nenvs = models[0].nenvs
        self.num_players = num_players
        self.models = models
        self.num_actions = models[0].num_actions
        self.use_lstm = False if models[0].init_state is None else True
        self.wait_rewards = wait_rewards
        if train_players is None:
            self.train_players = np.arange(0, num_players)
        # reset env and initialize observations
        self.players = [MAPlayer(i, models[i],) for i in range(self.num_players)]
        self.current_player = 0
        self.ep_num = 0
        self.schedule = train_schedule
        self.updates_wait = updates_wait
        self.reset()
        self.total_steps = 0
       

    
    def gen_noise_for_stepping(self, p, noisescale = 1):
        noise_ph_list = p.model.step_model.noise_list
        noise_val_list = []
        for noise_ph in noise_ph_list:
            noise_val_list.append((np.random.normal(0, noisescale, noise_ph.shape)))
        self.noise_val_list[p.num] = noise_val_list

                                   
    def reset(self):
        # resets Game, clears episodes stats. Should be ran after training data was collected.
        self.obs, _, self.legal_moves, self.ep_done, self.scores, self.ep_custom_rewards =\
        parse_timestep(self.env.reset())
        self.last_episodes_custom_rewards = {k : [] for k in self.ep_custom_rewards}
        self.prev_rewards = np.zeros((self.num_players, self.nenvs))
        self.prev_dones = np.ones((self.num_players, self.nenvs), dtype =  'bool')
        self.ep_stats = np.zeros((2, self.nenvs))
        self.ready_envs = set()
        self.last_episodes_reward = []
        self.last_episodes_length = []
        self.last_episodes_score = []
        self.train_counter = 0
        self.noise_val_list = {}
        if self.schedule is 'turn':
            self.cur_train_player = 0
            print('Player to train is %d' % self.cur_train_player)
        for p in self.players:
            self.gen_noise_for_stepping(p)
        
    def check_if_env_ready(self, i):
        # checks if env i collected enough data for training 
        for p in self.players:
            if len(p.history_buffer[i]['values']) < p.nsteps + 1:
                return False
        
        self.ready_envs.add(i)
        return True
            
    def play_turn(self, temp = 1):
        # current player plays one turn
        player = self.players[self.current_player]
        actions = player.step(self.legal_moves, self.obs, self.prev_dones[self.current_player], 
                              self.prev_rewards[self.current_player],self.noise_val_list[player.num], temp)
        
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
        
        self.current_player = (self.current_player + 1) % self.num_players
        self.obs, self.legal_moves, self.ep_done = obs, legal_moves, dones
        
    def play_untill_train(self, noisescale = 1, temp = 1):
        # runs the game untll all envs collect enough data for training
        #self.reset()
        for p in self.players:
            self.gen_noise_for_stepping(p, noisescale)
        self.ready_envs = set()
        self.last_episodes_reward = []
        self.last_episodes_score = []
        self.last_episodes_length = []
        self.last_episodes_custom_rewards = {k : [] for k in self.ep_custom_rewards}
        
        while len(self.ready_envs) < self.nenvs:
            self.play_turn(temp)
            for j, d in enumerate(self.ep_done):
                if d:
                    self.finish_episode(j)
                    
                self.check_if_env_ready(j)
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
        for p in self.players:
            if p.waiting[j]:
                p.history_buffer[j]['rewards'].append(self.prev_rewards[p.num][j])
                p.history_buffer[j]['dones'].append(self.prev_dones[p.num][j])
            p.waiting[j] = False
            

    
    def train_model(self, k, target_kl):
        policy_losses, value_losses, policy_entropies = defaultdict(list), defaultdict(list), defaultdict(list)
        train_steps = defaultdict(list)
        #action_distributions = {}
        state_probs = [[] for _ in range(self.num_players)]
        for p in self.players:
            
            (mb_obs,  mb_actions, mb_probs, mb_neglogps, mb_legal_moves, mb_values,
             mb_rewards, mb_masks, mb_dones, mb_states, mb_states_v, mb_noise) = self.collect_data(p.num)
            #action_distributions[p.num] = mb_actions
            state_probs[p.num] = (mb_obs, mb_actions, mb_probs, mb_legal_moves)
            if self.schedule is 'turn':
                if self.cur_train_player != p.num:
                    continue
            if p.num not in self.train_players:
                continue
            p.model.sess.run(p.model.increment_updates)
            for i in range(k):
                policy_loss, value_loss, policy_entropy, kl, probs = p.model.train(mb_obs, 0.2, mb_neglogps, 
                                                                                   mb_states, mb_rewards, 
                                                                                   mb_masks, mb_actions, 
                                                                                   mb_values, mb_states_v,
                                                                                   mb_legal_moves, mb_noise)
                if kl > target_kl:
                    break
            policy_losses[p.num] = policy_loss
            value_losses[p.num] = value_loss
            policy_entropies[p.num] = policy_entropy
            train_steps[p.num] = i + 1
            
            
        self.train_counter += 1
        if self.schedule is 'turn':
            if self.train_counter % self.updates_wait == 0:
                self.train_counter = 0
                self.cur_train_player = (self.cur_train_player + 1) % self.num_players
                print('Player to train is %d' % self.cur_train_player)
                while self.cur_train_player not in self.train_players:
                    self.cur_train_player = (self.cur_train_player + 1) % self.num_players
                    
        return state_probs, policy_losses, value_losses, policy_entropies, train_steps
       
    def collect_data(self, pnum):
        # collects data for training, clears all history buffers
        p = self.players[pnum]
        (mb_obs, mb_actions, mb_probs, mb_neglogps, mb_values, mb_rewards, mb_dones, 
         mb_masks, mb_states, mb_states_v, mb_legal_moves) = p.get_training_data()
        
        last_values = np.array(mb_values)[:, -1]
        mb_values = np.array(mb_values)[:, :-1]
        if not p.use_lstm:
            mb_states = []
            mb_states_v = []
        
        p.reset()
        mb_noise = self.noise_val_list[p.num]
        mb_actions = np.concatenate(mb_actions, 0)
        mb_probs = np.concatenate(mb_probs, 0)
        mb_neglogps = np.concatenate(mb_neglogps, 0)
        mb_masks = np.concatenate(mb_masks, 0)
        mb_obs = np.concatenate(mb_obs, 0)
        mb_legal_moves = np.concatenate(mb_legal_moves, 0)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        mb_dones = np.array(mb_dones)
        mb_rewards = np.array(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(p.model.nsteps)):
            nextnonterminal = 1.0 - mb_dones[:, t]
            if t == p.model.nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[:, t + 1]
            
            delta = mb_rewards[:, t] + p.model.gamma * nextvalues * nextnonterminal - mb_values[:, t]
            mb_advs[:, t] = lastgaelam = delta + p.model.gamma * .95 * nextnonterminal * lastgaelam
            
        mb_returns = mb_advs + mb_values
        mb_dones = np.concatenate(mb_dones, 0)
        mb_rewards = np.concatenate(mb_returns, 0)
        mb_values = np.concatenate(mb_values, 0)
        if not p.use_lstm:
            mb_states = None
            mb_staes_v = None
        return (mb_obs, mb_actions, mb_probs, mb_neglogps, mb_legal_moves, mb_values, mb_rewards,
               mb_masks, mb_dones, mb_states, mb_states_v, mb_noise)