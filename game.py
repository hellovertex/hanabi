import numpy as np
from collections import defaultdict
from tf_agents_lib import parallel_py_environment
from bad_agent import Player
from custom_environment.pub_mdp import PubMDP
from custom_environment.pubmdp_env_wrapper import PubMDPWrapper


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
    #beliefs_prob_dict = ts[3]['beliefs_prob_dict']
    score = ts[3]['score']
    custom_rewards = ts[3]['custom_rewards']
    return obs, rewards, legal_moves, dones, score, custom_rewards, None#beliefs_prob_dict


def _load_hanabi_pub_mdp(game_config):
    assert isinstance(game_config, dict)
    env = PubMDP(game_config)
    if env is not None:
        return PubMDPWrapper(env)
    return None


def load_env(env_config, num_envs):
    return parallel_py_environment.ParallelPyEnvironment(
        [lambda: _load_hanabi_pub_mdp(env_config)] * num_envs
    )


def load_specs(config):
    env = load_env(config, num_envs=1)
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    num_actions = action_spec.maximum + 1 - action_spec.minimum
    obs_size = obs_spec['state'].shape[0]

    print(f'GAME PARAMETERS: \n  observation length = {obs_size}\n  number of actions = {num_actions}\n')

    return obs_size, num_actions


class Game():
    def __init__(self, num_players, num_envs, env_config, wait_rewards=True):
        # create env
        self.env = load_env(env_config, num_envs)
        obs_size, num_actions = load_specs(env_config)
        self.observation_size = obs_size
        self.num_actions = num_actions

        # save params
        self.num_envs = num_envs
        self.num_players = num_players
        self.wait_rewards = wait_rewards
        self.players = [Player(i, num_envs) for i in range(self.num_players)]
        self.total_steps = 0
        self.reset()

    def reset(self, rewards_config=None, ):

        self.obs, _, self.legal_moves, self.ep_done, self.scores, self.ep_custom_rewards, self.beliefs_prob_dict = \
            parse_timestep(self.env.reset(rewards_config))
        self.current_player = 0
        self.steps_per_player = np.zeros(self.num_players)
        self.prev_rewards = np.zeros((self.num_players, self.num_envs))
        self.prev_dones = np.ones((self.num_players, self.num_envs), dtype='bool')
        self.ep_stats = np.zeros((2, self.num_envs))
        self.ready_envs = set()
        self.last_episodes_stats = defaultdict(list)
        for p in self.players:
            p.reset()
        self.prev_actions = -np.ones(self.num_envs, dtype='int')
        self.prev_obs = np.zeros((self.num_envs, 171))

    def play_turn(self, ):
        # current player plays one turn
        player = self.players[self.current_player]
        # self.obs contains the public belief
        actions, probs, alogps, values = player.step(self.legal_moves, self.obs, self.prev_dones[self.current_player],
                                                     self.prev_rewards[self.current_player],
                                                     self.prev_actions, self.prev_obs, self.beliefs_prob_dict)

        self.prev_actions = actions
        self.prev_obs = np.copy(self.obs)
        self.steps_per_player[player.num] += 1
        # todo ts must contain augmented_observations returned from pub_mdp.PubMDP.step
        ts = self.env.step(np.array(actions))
        # todo new: parse_timestep could also contain the publicbelief update
        obs, rewards, legal_moves, dones, scores, custom_rewards, beliefs_prob_dict = parse_timestep(ts)

        for k in self.ep_custom_rewards:
            self.ep_custom_rewards[k] = self.ep_custom_rewards[k] + custom_rewards[k]

        self.scores = scores

        self.total_steps += self.num_envs
        if self.wait_rewards:
            self.prev_rewards[self.current_player] = 0
            self.prev_rewards += rewards
        else:
            self.prev_rewards[self.current_player] = rewards
        self.ep_stats += np.array([rewards, [1] * self.num_envs])

        self.prev_dones[self.current_player] = 0
        self.prev_dones += dones

        self.current_player = (self.current_player + 1) % self.num_players
        self.obs, self.legal_moves, self.ep_done = obs, legal_moves, dones

    def eval_results(self, episodes_per_env=6, noisescale=1, ):
        # runs the game untll gall envs collect enough data for training
        self.reset()
        self.players[0].generate_noise(noisescale=noisescale)
        self.players[1].generate_noise(self.players[0].noise_val_list, noisescale)
        episodes_done = np.zeros(self.num_envs)
        record_env = np.ones(self.num_envs, dtype='bool')
        self.last_episodes_stats = defaultdict(list)

        while min(episodes_done) < episodes_per_env:
            self.play_turn()
            for j, d in enumerate(self.ep_done):
                if d:
                    if episodes_done[j] >= episodes_per_env:
                        record_env[j] = False
                    self.finish_episode(j, record_env[j])
                    episodes_done[j] += 1

        all_data = self.collect_data()
        obses = all_data[0]
        actions = all_data[1]
        probs = all_data[2]
        return self.last_episodes_stats

    def play_untill_train(self, nepisodes=90, nsteps=None, noisescale=1):
        # runs the game untll gall envs collect enough data for training
        # self.reset()

        self.last_episodes_stats = defaultdict(list)
        self.players[0].generate_noise(noisescale=noisescale)
        self.players[1].generate_noise(self.players[0].noise_val_list, noisescale)

        self.first_player = self.current_player
        self.steps_per_player = np.zeros(self.num_players)

        episodes_done = 0
        steps_done = np.zeros(self.num_players)
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

    def finish_episode(self, j, record=True):
        # Updates last reward for env j. Moves data froms players' episode buffer to history buffer
        R, L = self.ep_stats[:, j]
        self.prev_obs[j] = -1
        self.prev_actions[j] = -1
        if record:
            for k in self.ep_custom_rewards:
                self.last_episodes_stats[k].append(self.ep_custom_rewards[k][j])
                self.ep_custom_rewards[k][j] = 0
            self.last_episodes_stats['rewards'].append(R)
            self.last_episodes_stats['lengths'].append(L)
            self.last_episodes_stats['scores'].append(self.scores[j])

        self.ep_done[j] = False
        self.ep_stats[:, j] = 0, 0

        for p in self.players:
            if p.waiting[j]:
                p.history_buffer[j]['rewards'].append(self.prev_rewards[p.num][j])
                p.history_buffer[j]['dones'].append(self.prev_dones[p.num][j])

            p.waiting[j] = False

    def collect_data(self, player_nums='all'):
        if player_nums == 'all':
            player_nums = list(range(self.num_players))
        (mb_obses, mb_obses_ext, mb_actions, mb_probs, mb_alogps, mb_legal_moves, mb_values,
         mb_returns, mb_dones, mb_masks, mb_states, mb_states_v, mb_noise) = \
            [], [], [], [], [], [], [], [], [], [], [], [], []
        ts_take = int(np.min(self.steps_per_player)) - 1

        # print('TS TAKE', ts_take)
        for p in self.players:
            if p.num not in player_nums:
                continue

            (obses, obses_ext, actions, probs, alogps, legal_moves, values,
             returns, dones, masks, states, states_v, noise) = p.get_training_data(ts_take)
            mb_obses.append(obses)
            mb_obses_ext.append(obses_ext)
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
            # print('concating states')
            mb_states = np.concatenate(mb_states, 1)
            if states_v[0] is not None:
                mb_states_v = np.concatenate(mb_states_v, 1)
        # print('Game output states shape', np.shape(states), np.shape(states_v))
        return (np.concatenate(mb_obses), np.concatenate(mb_obses_ext), np.concatenate(mb_actions),
                np.concatenate(mb_probs), np.concatenate(mb_alogps), np.concatenate(mb_legal_moves),
                np.concatenate(mb_values), np.concatenate(mb_returns), np.concatenate(mb_dones),
                np.concatenate(mb_masks), np.array(mb_states), np.array(mb_states_v), np.concatenate(mb_noise))
