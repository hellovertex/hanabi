from collections import deque, defaultdict
import numpy as np
from PyHanabiWrapper import *
from tf_agents.environments import parallel_py_environment
from CardCounting import *
from .gamesettings import *
from .beliefs import *

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
    return obs, rewards, legal_moves, dones
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


    
class Player():
    def __init__(self, num, model, mini_model):
        # params
        self.num = num
        self.model = model
        self.mini_model = mini_model
        self.gamma = model.gamma
        self.nsteps = model.nsteps
        self.nenvs = model.nenvs
        self.use_lstm = False if model.init_state is None else True
        self.reset()
        self.reset_beliefs()
        
    def reset(self):
        # resets history, waiting status and states  
        self.history_buffer =  [defaultdict(list) for _ in range(self.nenvs)]
        
        self.waiting = [False for _ in range(self.nenvs)]
        self.states = self.model.init_state
        self.states_v = self.model.init_state_v
        if not self.use_lstm:
            self.states = None
            self.states_v = None
            
    def reset_beliefs(self):
        self.beliefs = [Belief() for _ in range(self.model.nenvs)]
    
    def step(self, legal_moves, obs, prev_dones, prev_rewards, prev_actions, noise = None, ) :
        # checks if player waits for reward from previous step, updates it if so
        # computes next actions, values, states and states_v, updates episode buffer
        for i, (w, p_r, p_d) in enumerate(zip(self.waiting, prev_rewards, prev_dones)):
            if w:
                self.history_buffer[i]['rewards'].append(p_r)
                self.history_buffer[i]['dones'].append(p_d)
                
        all_hands = self.beliefs[0].all_hands
        #print('P', self.num)
        obs_with_beliefs = []
        for obs_vec, lm_vec, b, prev_a, prev_d in zip(obs, legal_moves, self.beliefs, prev_actions, prev_dones):
            if prev_d:
                b.reset()
            else:
                b.process_observation(obs_vec)
            MM_output = self.mini_model.compute_action_probs(all_hands, obs_vec, legal_moves)[:, prev_a]
            beliefs = b.compute_hand_beliefs(np.exp(MM_output))
            #print("BELIEFS", beliefs)
            obs_with_beliefs.append(np.concatenate([obs_vec, beliefs]))

        actions, probs, neglogps, values, states, states_v = self.model.step(obs_with_beliefs, self.states,
                                                                             prev_dones, self.states_v,
                                                                             legal_moves, noise)
        values = values[:, 0]
        for i in range(self.nenvs):
            self.history_buffer[i]['obses'].append(obs[i])
            self.history_buffer[i]['obses_with_beliefs'].append(obs_with_beliefs[i])
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
        obses, obses_with_beliefs ,actions,probs,neglogps,values, rewards, dones,\
        masks, states, states_v, noise, legal_moves = \
        [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(self.nenvs):
            obses.append(self.history_buffer[i]['obses'][ : self.model.nsteps])
            obses_with_beliefs.append(self.history_buffer[i]['obses_with_beliefs'][ : self.model.nsteps])
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

        if self.use_lstm:
            states = np.swapaxes(states, 0, 1)
            if states_v[0] is not None:
                states_v = np.swapaxes(states_v, 0, 1 )
        self.history_buffer = [defaultdict(list) for _ in range(self.nenvs)]
        return (obses, obses_with_beliefs, actions, probs, neglogps, values, 
                rewards, dones, masks, states, states_v, legal_moves)
    
class Game():
    def __init__(self, num_players, model, mini_model, load_env, clip_func):
        
        # create env
        self.env = parallel_py_environment.ParallelPyEnvironment([load_env] * model.nenvs)
       
        # save params
        self.nenvs = model.nenvs
        self.num_players = num_players
        self.model = model
        self.mini_model = mini_model
        self.num_actions = model.num_actions
        self.use_lstm = False if model.init_state is None else True
        self.clip_func = clip_func
        # reset env and initialize observations
        self.players = [Player(i, model, mini_model) for i in range(self.num_players)]
        self.current_player = 0
        self.ep_num = 0
        self.reset()
        self.total_steps = 0
    
    def gen_noise_for_stepping(self, noisescale = 1):
        noise_ph_list = self.model.step_model.noise_list
        noise_val_list = []
        for noise_ph in noise_ph_list:
            noise_val_list.append((np.random.normal(0, noisescale, noise_ph.shape)))
        self.noise_val_list = noise_val_list

        
                                   
    def reset(self):
        # resets Game, clears episodes stats. Should be ran after training data was collected.
        self.obs, _, self.legal_moves, self.ep_done = parse_timestep(self.env.reset())
        self.prev_actions = -np.ones(self.nenvs, dtype = 'int')
        self.prev_rewards = np.zeros((self.num_players, self.nenvs))
        self.prev_dones = np.ones((self.num_players, self.nenvs), dtype =  'bool')
        self.ep_stats = np.zeros((3, self.nenvs))
        self.ready_envs = set()
        self.last_episodes_reward = []
        self.last_episodes_clipped_reward = []
        self.last_episodes_length = []
        self.game_states = []
        self.gen_noise_for_stepping()
        
    def check_if_env_ready(self, i):
        # checks if env i collected enough data for training 
        for p in self.players:
            if len(p.history_buffer[i]['values']) < p.nsteps + 1:
                return False
        
        self.ready_envs.add(i)
        return True
            
   
    def eval_results(self, nepisodes, noisescale = 1):
        self.reset()
        self.gen_noise_for_stepping(noisescale)
        self.last_episodes_reward = []
        self.last_episodes_clipped_reward = []
        self.last_episodes_length = []
        
        nep = 0
        while nep < nepisodes:
            self.play_turn()
            for j, d in enumerate(self.ep_done):
                if d:
                    self.finish_episode(j)
                    nep += 1
        return self.last_episodes_clipped_reward, self.last_episodes_reward, self.last_episodes_length
            
    def play_turn(self,):
        # current player plays one turn
        player = self.players[self.current_player]
        
        actions = player.step(self.legal_moves, self.obs, self.prev_dones[self.current_player], 
                              self.prev_rewards[self.current_player], self.prev_actions, self.noise_val_list)
        

        ts = self.env.step(np.array(actions))
        obs, rewards, legal_moves, dones = parse_timestep(ts)
        for env_id, a in enumerate(actions):
            a_type = get_action_type(a)
            if a_type == 0:
                player.beliefs[env_id].process_played_card(obs[env_id], offset = 1)
                
        if self.clip_func is not None:
            clipped_reward = self.clip_func(rewards)
        else:
            clipped_reward = rewards
        # update stats
        self.total_steps += self.nenvs
        self.prev_actions = actions
        self.prev_rewards[self.current_player] = 0
        self.prev_rewards += clipped_reward
        self.ep_stats += np.array([rewards, clipped_reward, [1] * self.nenvs])
        self.prev_dones[self.current_player] = 0
        self.prev_dones += dones
        
        self.current_player = (self.current_player + 1) % self.num_players
        self.obs, self.legal_moves, self.ep_done = obs, legal_moves, dones

    def play_untill_train(self, noisescale = 1):
        # runs the game untll gall envs collect enough data for training
        #self.reset()
        self.gen_noise_for_stepping(noisescale)
        self.ready_envs = set()
        self.last_episodes_reward = []
        self.last_episodes_clipped_reward = []
        self.last_episodes_length = []
        while len(self.ready_envs) < self.nenvs:
            self.play_turn()
            for j, d in enumerate(self.ep_done):
                if d:
                    self.finish_episode(j)
                    
                self.check_if_env_ready(j)
        return self.last_episodes_clipped_reward, self.last_episodes_reward, self.last_episodes_length
    def finish_episode(self, j):
        # Updates last reward for env j. Moves data froms players' episode buffer to history buffer
        R, R_clipped, L = self.ep_stats[:, j]
        self.last_episodes_clipped_reward.append(R_clipped)
        self.last_episodes_reward.append(R)
        self.last_episodes_length.append(L)
        self.ep_num += 1
        self.ep_done[j] = False
        self.ep_stats[:, j] = 0,0,0
        for p in self.players:
            if p.waiting[j]:
                p.history_buffer[j]['rewards'].append(self.prev_rewards[p.num][j])
                p.history_buffer[j]['dones'].append(self.prev_dones[p.num][j])
            p.waiting[j] = False
            p.beliefs[j].reset()
    
    def collect_data(self):
        # collects data for training, clears all history buffers
        mb_obs, mb_actions, mb_neglogps, mb_legal_moves, mb_values, mb_rewards, mb_masks, mb_dones, mb_noise = \
        [], [], [], [], [], [], [], [], []
        mb_states, mb_states_v =  [], []
        mb_probs = []
        mb_obs_ext = []
        last_values = []
        for p in self.players:
            obses, obses_ext, actions, probs, neglogps, values, rewards, dones, masks, states, states_v, legal_moves = \
            p.get_training_data()
            mb_obs.append(obses)
            mb_obs_ext.append(obses_ext)
            mb_actions.append(actions)
            mb_probs.append(probs)
            mb_neglogps.append(neglogps)
            mb_legal_moves.append(legal_moves)
            mb_values.append(np.array(values)[:, :-1])
            last_values.append(np.array(values)[:, -1])
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            mb_masks.append(masks)
            if self.use_lstm:
                mb_states.append(states)
                mb_states_v.append(states_v)
            p.reset()
        mb_noise = self.noise_val_list
        mb_actions = np.concatenate(mb_actions, 0)
        mb_actions = np.concatenate(mb_actions, 0)
        mb_neglogps = np.concatenate(mb_neglogps, 0)
        mb_neglogps = np.concatenate(mb_neglogps, 0)
        mb_values = np.concatenate(mb_values, 0)
        last_values = np.concatenate(last_values, 0)
        mb_masks = np.concatenate(mb_masks, 0)
        mb_masks = np.concatenate(mb_masks, 0)
        mb_dones = np.concatenate(mb_dones, 0)
        
        mb_obs = np.concatenate(mb_obs, 0)
        mb_obs = np.concatenate(mb_obs, 0)
        mb_obs_ext = np.concatenate(mb_obs_ext, 0)
        mb_obs_ext = np.concatenate(mb_obs_ext, 0)
        mb_probs = np.concatenate(mb_probs, 0)
        mb_probs = np.concatenate(mb_probs, 0)
        mb_legal_moves = np.concatenate(mb_legal_moves, 0)
        mb_legal_moves = np.concatenate(mb_legal_moves, 0)
        
        mb_rewards = np.concatenate(mb_rewards, 0)

        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.model.nsteps)):
            nextnonterminal = 1.0 - mb_dones[:, t]
            if t == self.model.nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[:, t + 1]
            
            delta = mb_rewards[:, t] + self.model.gamma * nextvalues * nextnonterminal - mb_values[:, t]
            mb_advs[:, t] = lastgaelam = delta + self.model.gamma * .95 * nextnonterminal * lastgaelam
            
        mb_returns = mb_advs + mb_values
        mb_dones = np.concatenate(mb_dones, 0)
        mb_rewards = np.concatenate(mb_returns, 0)
        mb_values = np.concatenate(mb_values, 0)
        if self.use_lstm:
            mb_states = np.concatenate(mb_states, 1)
            if self.model.train_model.v_net_type == 'copy':
                mb_states_v = np.concatenate(mb_states_v, 1)
        else:
            mb_states = None
            mb_staes_v = None
        return( mb_obs, mb_obs_ext, mb_actions, mb_probs, mb_neglogps, mb_legal_moves, mb_values, mb_rewards,
               mb_masks, mb_dones, mb_states, mb_states_v, mb_noise)
    
    def train_model(self, k, target_kl):
        (obs, obses_with_beliefs, actions, probs, neglogps, legal_moves, values,
         rewards, masks, dones, states, states_v, noise) = self.collect_data()
        data = (obses_with_beliefs, actions, neglogps, legal_moves, values,
                rewards, masks, dones)
        inds = np.arange(len(rewards))
        nenvs = self.model.nenvs * self.model.nplayers
        envsperbatch = nenvs // self.model.nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * self.model.nsteps).reshape(nenvs, self.model.nsteps)
        state_probs = (obs, actions, probs, legal_moves)
        for i in range(k):
            np.random.shuffle(envinds)
            p_losses, v_losses, p_ents = [], [], []
            mean_kl = []
            if self.use_lstm:
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start : end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    data_mb = (arr[mbflatinds] for arr in data)
                    (mb_obs, mb_actions, mb_neglogps, mb_legal_moves, mb_values,
                     mb_rewards, mb_masks, mb_dones) = data_mb
                    mb_noise = noise
                    mb_states = states[:, mbenvinds, :]
                    if self.model.train_model.v_net_type == 'copy':
                        mb_states_v = states_v[:, mbenvinds, :]
                    else:
                        mb_states_v = mb_states

                    policy_loss, value_loss, policy_entropy, kl, probs = self.model.train(mb_obs, 0.2, mb_neglogps, 
                                                                                     mb_states, mb_rewards, 
                                                                                     mb_masks, mb_actions, 
                                                                                     mb_values, mb_states_v,
                                                                                     mb_legal_moves, mb_noise)
                    
                    p_losses.append(policy_loss)
                    v_losses.append(value_loss)
                    p_ents.append(policy_entropy)
                    mean_kl.append(kl)
            else:
                nbatch_train = len(rewards) // self.model.nminibatches
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, len(rewards), nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start : end]
                    data_mb = (arr[mbinds] for arr in data)
                    (mb_obs, mb_actions, mb_neglogps, mb_legal_moves, mb_values,
                    mb_rewards, mb_masks, mb_dones) = data_mb
                    mb_states = states
                    mb_states_v = states_v
                    mb_noise = noise
                    policy_loss, value_loss, policy_entropy, kl, probs = self.model.train(mb_obs, 0.2, mb_neglogps, 
                                                                                     mb_states, mb_rewards, 
                                                                                     mb_masks, mb_actions, 
                                                                                     mb_values, mb_states_v,
                                                                                     mb_legal_moves, mb_noise)
                    
                    p_losses.append(policy_loss)
                    v_losses.append(value_loss)
                    p_ents.append(policy_entropy)
                    mean_kl.append(kl)
                
            mean_kl = np.mean(mean_kl)
            if mean_kl > target_kl:
                break
        self.model.sess.run(self.model.increment_updates)
        return state_probs, np.mean(p_losses), np.mean(v_losses), np.mean(p_ents), i + 1 
    