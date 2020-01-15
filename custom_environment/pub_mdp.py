import numpy as np
from scipy.stats import rv_discrete
from tf_agents.trajectories.policy_step import CommonFields
from tf_agents.trajectories.time_step import TimeStep, StepType
from hanabi_learning_environment import pyhanabi as pyhanabi, rl_env
from hanabi_learning_environment.pyhanabi import color_char_to_idx
from hanabi_learning_environment.pyhanabi import HanabiMoveType
MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]
COLOR_CHAR = ["R", "Y", "G", "W", "B"]  # consistent with hanabi_lib/util.cc


class PublicAgent(object):
    def __init__(self, env_config, observation_size, hand_size, alpha=.1):
        self.env_config = env_config
        self.observation_size = observation_size
        self.alpha = alpha

        self.B_0 = None
        self.BB = None
        self.V1 = None
        self.V2 = None
        self._episode_ended = False
        self.last_time_step = None

        # pubmdp stats
        self.num_colors = env_config['colors']
        self.num_ranks = env_config['ranks']
        self.num_players = env_config['players']
        self.hand_size = hand_size
        self.num_slots = self.num_players * self.hand_size

        self.num_cards = self.num_colors * self.num_ranks + 1
        self.hint_mask = np.ones((self.num_slots, self.num_cards))
        self.hint_mask[:, -1] = 0
        self.bits_per_card = self.num_colors * self.num_ranks
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.candidate_counts = np.append(self.candidate_counts, 0)
        hint_mask_flattened = self.hint_mask.flatten()
        self.debug_utils_counter = 0
        self.len_public_features = len(hint_mask_flattened) + len(self.candidate_counts)
        self.len_s_bad = self.len_public_features + len(hint_mask_flattened)

        # utils
        self.util_mask_slots_hinted = list()
        self.deck_is_empty = False

    def reset(self):
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.candidate_counts = np.append(self.candidate_counts, 0)  # 1 if card is missing in last turn
        self.hint_mask = np.ones((self.num_players * self.hand_size, self.num_colors * self.num_ranks + 1))
        self.hint_mask[:, -1] = 0
        self.util_mask_slots_hinted = list()
        self.public_features = np.concatenate((self.candidate_counts, self.hint_mask.flatten()))
        self.deck_is_empty = False
        self.B_0 = None
        self.BB = None
        self.V1 = None
        self.V2 = None
        self._episode_ended = False
        self.last_time_step = None
        self.debug_utils_counter = 0

    def _reset_slot_hint_mask(self, last_move):
        """ After a card has been played, its corresponding row in the hint_mask must be reset """
        row = last_move.move().card_index()
        if row in self.util_mask_slots_hinted:  # todo check if this notably decreases performance
            self.util_mask_slots_hinted.remove(row)
        self.hint_mask[row, :] = 1
        self.hint_mask[:, self.candidate_counts == 0] = 0

    def _get_idx_candidate_count(self, last_move):
        """ Returns for PLAY or DISCARD moves, the index of the played card with respect to self.candidate_counts
        :arg
            last_move: a pyhanabi.HanabiHistoryItem containing the last non deal move
        :returns
            cards_count_idx: the index of the played/discarded card in self.candidate_counts
        """
        assert last_move.move().type() in [HanabiMoveType.PLAY, HanabiMoveType.DISCARD]
        color = last_move.color()
        rank = last_move.rank()
        assert ((color * self.num_ranks) + rank) < len(self.candidate_counts)
        return (color * self.num_ranks) + rank

    def reduce_card_candidate_count(self, last_move):
        """
        When a card has been played or discarded, decrement in self.candidate_counts
        :returns card_candidate_idx: the index of the played/discarded card w.r.t. self.candidate_counts
         """

        card_candidate_idx = self._get_idx_candidate_count(last_move)
        assert card_candidate_idx in [i for i in range(len(self.candidate_counts))]
        assert self.candidate_counts[card_candidate_idx] > 0, f'card_candidate_idx = {card_candidate_idx}\ncandidate counts = {self.candidate_counts}'

        self.candidate_counts[card_candidate_idx] -= 1

        return card_candidate_idx

    @staticmethod
    def _get_last_non_deal_move(last_moves):
        """ Returns last non deal move
        Args:
            last_moves: [pyhanabi.HanabiHistoryItem()]
        Returns:
             last_non_deal_move: pyhanabi.HanabiHistoryItem()
            """
        assert last_moves is not None
        assert isinstance(last_moves, list)

        if len(last_moves) > 0:
            # print('hello', type(last_moves[0]))
            # assert isinstance(last_moves[0], pyhanabi.HanabiHistoryItem)
            pass
        i = 0
        last_move = None
        while True:
            if len(last_moves) <= i:
                break
            if last_moves[i].move().type() != HanabiMoveType.DEAL:
                last_move = last_moves[i]
                break
            i += 1

        return last_move

    def _update_hint_mask(self, last_move):
        """
        Updates the hint_mask given last_move

        Args fetched from self._get_info_reveal_move(last_move):
             color: 0-indexed color corresponding to ["R", "Y", "G", "W", "B"]
             rank: 0-indexed rank
             agent_index: absolute position of agent
             cards_revealed: indices of cards touched by a clue, e.g. [0,2,3]
        ---------------------------------------------------------------------
        e.g. for an initial hintmask
        self.hint_mask = [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
                          [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
                          [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
                          [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]]
        and REVEAL_COLOR: 0 to player 0, e.g.
            color, rank, agent_idx, cards_revealed = 0, -1, 0, [0,1]

        the function would update self.hint_mask to
            [[1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
             [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]

        When REVEAL_COLOR: 1 however, the result will be
             [[0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0.]
              [0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0.]
              [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
        ----------------------------------------------------
        Similarly, when target is player 1, the results will be
            [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
             [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]]
        and
            [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
             [0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        for REVEAL_COLOR: 0 and REVEAL_RANK: 1 respectively.
         """
        move_type = last_move.move().type()
        assert move_type in [HanabiMoveType.REVEAL_RANK, HanabiMoveType.REVEAL_COLOR]
        color, rank, target_agent_idx, target_cards_idx = self._get_info_reveal_move(last_move)

        # Slots are rows in self.hint_mask corresponding to the cards touched by a REVEAL_XYZ move
        slots = [target_agent_idx * self.hand_size + slot for slot in target_cards_idx]

        # set all other touched slots to 0, unless they have been hinted already
        mask_untouched = [i for i in slots if i not in self.util_mask_slots_hinted]
        self.hint_mask[mask_untouched] = 0

        # add rows that were hinted
        [self.util_mask_slots_hinted.append(slot) for slot in slots]

        if move_type == HanabiMoveType.REVEAL_COLOR:
            col_idx = color * self.num_ranks
            self.hint_mask[slots, col_idx:col_idx + self.num_ranks] = 1
        elif move_type == HanabiMoveType.REVEAL_RANK:
            cols = [rank + self.num_ranks * c for c in range(self.num_colors)]
            for rank in cols:
                self.hint_mask[slots, rank] = 1
        else:
            raise ValueError

    def get_abs_agent_idx(self, last_move):
        """ Computes absolute agent position given target_offset and current_player_position,
        i.e. (agent_position + target_offset) % num_players
        Args:
            last_move: pyhanabi.HanabiHistoryItem containing information on the action last taken
        """
        return (last_move.player() + last_move.move().target_offset()) % self.num_players

    def _get_info_reveal_move(self, last_move):
        """ Returns color, rank, and agent_index for a given move of type REVEAL_XYZ
        Args:
            last_move: pyhanabi.HanabiHistoryItem containing information on the action last taken
        Returns:
            color, rank,  target_agent_idx, target_cards_idx: 0-based values,
            color is None or -1 for REVEAL_RANK moves and rank is None or -1 for REVEAL_COLOR moves
        """
        assert last_move.move().type() in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]
        target_cards_idx = last_move.card_info_revealed()
        color = last_move.move().color()
        rank = last_move.move().rank()
        target_agent_idx = self.get_abs_agent_idx(last_move)

        return color, rank, target_agent_idx, target_cards_idx

    def update_candidates_and_hint_mask(self, last_moves):
        """ Update public card knowledge given last move, More specifically
         For PLAY and DISCARD moves:
            - reduce card candidate count by 1
            - if last copy was played, set corresponding self.hint_mask columns to 0
            - reset hint_mask for corresponding card slot
         For REVEAL_COLOR and REVEAL_RANK moves
            - update self.hint_mask
         For DEAL moves:
            - skip
        Does nothing if there are no last_moves.
         """

        if last_moves:  # is False on Initialization, i.e. when last_moves is empty, i.e. when last_moves is False
            last_move = self._get_last_non_deal_move(last_moves)  # pyhanabi.HanabiHistoryItem()
            # PLAY and DISCARD moves
            if last_move.move().type() in [HanabiMoveType.PLAY, HanabiMoveType.DISCARD]:
                # Reduce card candidate count by 1 and if last copy was played, set corresponding hint_mask columns to 0
                card_candidate_idx = self.reduce_card_candidate_count(last_move)
                if self.candidate_counts[card_candidate_idx] == 0:
                    self.hint_mask[:, card_candidate_idx] = 0
                self._reset_slot_hint_mask(last_move)

            # REVEAL_COLOR and REVEAL_RANK moves
            elif last_move.move().type() in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]:
                self._update_hint_mask(last_move)
            else:
                raise ValueError

    @staticmethod
    def get_last_moves_from_obs(observations):
        """ observations contains all the hands of all the players """

        current_player = observations['current_player']
        obs = observations['player_observations'][current_player]
        return obs['pyhanabi'].last_moves()

    def compute_B0_belief(self):
        """ Computes initial public belief that only depends on card counts and hint mask. C.f. eq(12)
        Can be used for baseline agents as well as re-marginalization initialization

        e.g. for 2 players and 2 colors (5 ranks) at start of the game
        initial belief B0 [[3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 0 of player 0 --> B(f[0])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 1 of player 0 --> B(f[1])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 0 of player 1 --> B(f[2])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]] # corresponds to card 1 of player 1 --> B(f[3])

        """
        B0 = self.candidate_counts * self.hint_mask
        B0_normalized = B0 / np.sum(B0, axis=1, keepdims=True)
        # return B0  # broadcasts self.candidate_counts to rows of self.hint_mask
        return B0_normalized

    def uniform_likelihood(self):
        """ Returns uniform probabilities for likelihood of private features.
        These are used to initialize the bayesian belief on the first turn"""
        unnormalized = np.ones((self.num_slots, self.num_cards))
        return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)

    def compute_belief_at_convergence(self, B_0, k=5):
        """ Computes re-marginalized V1 beliefs C.f. eq(13) """
        # Re-marginalize and save to self.public-belief
        assert isinstance(B_0, np.ndarray)
        B_k = np.copy(B_0)
        B_kplus1 = np.zeros((self.num_players * self.hand_size, self.num_colors * self.num_ranks + 1))
        num_slots = B_k.shape[0]
        for _ in range(k):  # k << 10 suffices for until I improved implementation
            # iterate public_belief rows which is 20 iterations at most
            # for i, slot in enumerate(belief_v1):
            for i in range(num_slots):  # todo remove this loop
                B_kplus1[i] = self.candidate_counts - np.sum(
                    B_k[(np.arange(self.num_players * self.hand_size) != i)],  # pick all other rows
                    axis=0)  # sum across other slots
            B_k = B_kplus1 / np.sum(B_kplus1, axis=1, keepdims=True) * self.hint_mask

        # print(re_marginalized)
        return B_k

    def _remove_inconsistent_samples(self, samples, num_consistent_samples_to_return=3000):
        """
                Samples are taken from the posterior belief, B0 or V1, over the private features
                E.g.
                samples = [[5. 7. 8. ... 0. 5. 0.]  # 3000 samples for first card of first player
                           [0. 4. 6. ... 7. 0. 8.]  # 3000 samples for second card of first player
                           [2. 0. 8. ... 2. 2. 5.]
                           [3. 7. 8. ... 1. 5. 1.]]  # 3000 samples for last card of last player

                where the entries are card_numbers between 0 and (colors * ranks) and rows correspond to slots.

                A sample (column) is called consistent, iff the following conditions hold:
                i) The cards sampled are available according to self.candidate_counts  (not all copies already played)
                ii) The cards sampled are possible according to self.hint_mask (card is possible given history of hints)
                """
        # store indices of consistent sampled by their column index
        consistent_samples = list()
        idx = 0
        # currently this loop takes ~70 to 160 ms to run for 20k samples. todo remove this loop
        for sample in samples.T:  # transposing is fastest, as it just changes the np.strides
            sample_card_counts = np.bincount(sample, minlength=len(self.candidate_counts))
            reduced_counts = self.candidate_counts - sample_card_counts
            # i) check sample consistent with candidate count
            if not reduced_counts[reduced_counts < 0].size:
                pass
            else:
                continue
            # ii) check sample consistent with hint_mask
            for shape, val in np.ndenumerate(sample):
                if self.hint_mask[shape[0], val] == 0:
                    continue
            # if this code is reached, the sample is consistent
            consistent_samples.append(idx)
            if len(consistent_samples) == num_consistent_samples_to_return:
                break
            idx += 1
        return samples[:, consistent_samples]

    def sample_consistent_private_features_from_public_belief(self, num_samples=3000, distr_priv_features='V1'):
        """
        In order to compute the bayesian beliefs (c.f. eq. 14, eq. 15), we need to have likelihood samples of
        the private features. These are computed using samples from the public belief (c.f. eq. 8).
        This method returns n samples of private features, that are ensured to be consistent with the deck.
        """
        # We provide the option of sampling private features from either B0 or V1 by setting distr_priv_features
        # sample from the public belief (either self.B0) or (self.V1)
        assert distr_priv_features in ['B0', 'V1']
        belief = getattr(self, distr_priv_features)  # either self.V1 or self.B0
        sampled = np.zeros(shape=(self.num_players * self.hand_size, num_samples), dtype=np.int)

        # compute card indices
        x_i = np.array([card_index for card_index in range(self.num_ranks * self.num_colors)])
        # and their probabilities per slot per player
        px = belief / np.sum(belief, axis=1, keepdims=True)  # todo remove last 1
        # sample hands from the resulting distribution
        for (i_row, _), row in np.ndenumerate(belief):  # todo remove this loop, i.e. sample multivariate
            sampled[i_row,] = rv_discrete(values=(x_i, px[i_row, :-1])).rvs(size=num_samples)

        # For now we just compute a fixed number of samples and take the first 3000 consistent ones
        return self._remove_inconsistent_samples(sampled, num_consistent_samples_to_return=3000)

    def compute_likelihood_private_features(self, last_moves, seed=123):
        """ Given an observed action, it computes its probability using the public policy.
         This action in turn determines the likelihood of
        """
        assert last_moves  # raise Error when last_moves is empty, because then there is no action yet
        # Otherwise, generate 3000 consistent samples

        samples = self.sample_consistent_private_features_from_public_belief(num_samples=3000)

        self.debug_utils_counter += 1
        # and get the last action
        last_action = self._get_last_non_deal_move(last_moves)
        suitable_private_features = list()
        private_features_that_lead_to_different_action = list()

        for sample in samples.T:  # todo remove this loop (Cython || C++)
            if not self.last_time_step is None:
                if not self.public_policy is None:
                    cnum = 0
                    sampled_obs = self.last_time_step.observation['state']
                    for card in sample:
                        sampled_obs[cnum*self.bits_per_card:cnum*self.bits_per_card+self.bits_per_card] = self.abs_cnum_to_one_hot(card)
                        cnum += 1
                    obs = {'state': self.last_time_step.observation['state'],  # sampled_obs
                           'mask': self.last_time_step.observation['mask']}

                    # replace 0s until self.len_own_hand_vectorized with encoded sample
                    step_type = self.last_time_step.step_type
                    reward = self.last_time_step.reward
                    discount = self.last_time_step.discount
                    timestep = TimeStep(step_type, reward, discount, obs)
                    # compute forward passes using ts
                    # assume that action has been sampled using the public seed
                    # print(f'timestep after replacement = {timestep}')
                    # todo maybe enable eager mode for tf
                    policy_step = self.public_policy.action(self.last_time_step)
                    action = policy_step.action
                    log_prob = getattr(policy_step.info, CommonFields.LOG_PROBABILITY)  # maybe xD
                    # print(f'logprob is {log_prob}')
                    if action == last_action:
                        # can compute the likelihood on the fly, without lists
                        suitable_private_features.append(log_prob)
                    # todo compute the actual values for the likelihood matrix, which is easy once the encoding is given

        return .1

    # LVL 2
    def compute_bayesian_belief(self, last_moves, k=1):
        """ Computes re-marginalized Bayesian beliefs. C.f. eq(14) and eq(15)"""
        # Compute basic bayesian belief
        ll = self.compute_likelihood_private_features(last_moves)  # uses self.last_time_step
        BB_k = self.B_0 * ll  # elementwise
        BB_kplus1 = np.zeros((self.num_players * self.hand_size, self.num_colors * self.num_ranks + 1))

        for _ in range(k):  # k << 10 suffices
            # iterate public_belief rows which is 20 iterations at most
            for i in range(self.num_slots):
                BB_kplus1[i] = self.candidate_counts - np.sum(
                    BB_k[np.arange(self.num_players * self.hand_size) != i], axis=0) * self.hint_mask[i, :] * ll
            BB_k = BB_kplus1 / np.sum(BB_kplus1, axis=1, keepdims=True)

        return BB_k

    def compute_public_state(self, observations, k=1):
        """ Computes public belief state s_bad shared by all agents"""
        if self.deck_is_empty:
            self.hint_mask[observations['current_player']:, -1] = 1
        last_moves = self.get_last_moves_from_obs(observations)  # may be empty
        self.update_candidates_and_hint_mask(last_moves)
        self.B_0 = self.compute_B0_belief()
        if not last_moves:
            self.V1 = self.B_0
            self.BB = self.B_0 * self.uniform_likelihood()  # on first turn, there was no action
        else:
            self.V1 = self.compute_belief_at_convergence(self.B_0)
            self.BB = self.compute_bayesian_belief(last_moves)
        self.V2 = (1 - self.alpha) * self.BB + self.alpha * self.V1
        self.public_features = np.concatenate(  # Card-counts and hint-mask
            (self.candidate_counts, self.hint_mask.flatten())
        )

        s_bad = np.append(self.public_features, self.V2.flatten())
        # print(f'len inside compute public state = {len(self.public_features), len(self.V2.flatten())}')
        return {'B0': self.B_0,
                'V1': self.V1,
                'BB': self.BB,
                'V2': self.V2,
                'f_pub': self.public_features,
                'vectorized': s_bad}

    def update_belief(self, observations, actions=None, network=None):
        public_belief_state = self.compute_public_state(observations)
        s_bad_vectorized = public_belief_state['vectorized']
        for player_dict in observations['player_observations']:
            player_dict['vectorized'] = np.append(s_bad_vectorized, player_dict['vectorized'])
        # todo from observations get target observation via offsetting
        return [0]

    @staticmethod
    def initialize_belief(obs):
        return [0]


class PubMDP(object):
    """ Sits on top of the hanabi-learning-environment and is responsible for computation of beliefs (-updates) that
    are used to augment an agents observation to match the state definition of Pub-MDPs
    c.f. 'Bayesian action decoder for deep multi-agent reinforcement learning' (Foerster et al.)
    (https://arxiv.org/abs/1811.01458)"""

    def __init__(self, game_config, public_policy=None, alpha=.1, tf_sess=None, use_beliefs=False):
        """
            See definition of augment_observation() at the bottom of this file to understand how this class works
        """
        assert isinstance(game_config, dict)
        self.env = rl_env.HanabiEnv(game_config)
        self.alpha = alpha
        self.num_colors = game_config['colors']
        self.num_ranks = game_config['ranks']
        self.num_players = game_config['players']
        self.hand_size = self.env.game.hand_size()
        self.num_slots = self.num_players * self.hand_size
        self.num_cards = self.num_colors * self.num_ranks + 1  # add 1 for None-card in last round
        # C(f): 0-1 vector of length (colors * ranks) containing public card counts
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.candidate_counts = np.append(self.candidate_counts, 0)  # 1 if card is missing in last turn
        # HM: binary matrix, slot equals 1, if player could be holding card at given slot
        # i.e. hint_mask[i,j] == 1 <==> (i % hand_size)th card of (i\hand_size)th player equals card j
        self.hint_mask = np.ones((self.num_slots, self.num_cards))
        self.hint_mask[:, -1] = 0

        self.util_mask_slots_hinted = list()  # maintains rows of self.hint_mask that have received a hint
        self.deck_is_empty = False
        """
        e.g. for 2 players and 2 colors (5 ranks) at start of the game
        self.hint_mask = [[1 1 1 1 1 1 1 1 1 1 0]  # card 0 of player 0  --> HM[f[0]] 
                          [1 1 1 1 1 1 1 1 1 1 0]  # card 1 of player 0  --> HM[f[1]]
                          [1 1 1 1 1 1 1 1 1 1 0]  # card 0 of player 1  --> HM[f[2]]
                          [1 1 1 1 1 1 1 1 1 1 0]  # card 1 of player 1  --> HM[f[3]]
        ] 
        """

        # these will be initialized on env.reset() call
        # correct? self.private_features = None  # 0-1 vector of length (num_players * hand_size) * (colors * ranks)
        hint_mask_flattened = self.hint_mask.flatten()
        self.public_features = np.concatenate(  # Card-counts and hint-mask
            (self.candidate_counts, hint_mask_flattened)
        )
        # re-loads public policy, in case the weights have changed
        self.public_policy = public_policy
        self.B_0 = None
        self.BB = None
        self.V1 = None
        self.V2 = None
        self._episode_ended = False
        self.last_time_step = None

        self.bits_per_card = self.num_colors * self.num_ranks
        #self.len_private_observation = (self.num_players - 1) * self.hand_size * self.bits_per_card + self.num_players
        #self.len_own_hand_vectorized = self.hand_size * self.bits_per_card
        # bits per player are set at the end of encoded handcards
        self.debug_utils_counter = 0
        self.len_public_features = len(hint_mask_flattened) + len(self.candidate_counts)
        self.len_s_bad = self.len_public_features + len(hint_mask_flattened)
        # print(f'INSIDE INIT, WE DETERMINED LEN_PUB_FEATURES = {self.len_public_features}')
        self.tf_sess = tf_sess
        self.use_beliefs = use_beliefs

    def _get_idx_candidate_count(self, last_move):
        """ Returns for PLAY or DISCARD moves, the index of the played card with respect to self.candidate_counts
        :arg
            last_move: a pyhanabi.HanabiHistoryItem containing the last non deal move
        :returns
            cards_count_idx: the index of the played/discarded card in self.candidate_counts
        """
        assert last_move.move().type() in [HanabiMoveType.PLAY, HanabiMoveType.DISCARD]
        color = last_move.color()
        rank = last_move.rank()
        assert ((color * self.num_ranks) + rank) < len(self.candidate_counts)
        return (color * self.num_ranks) + rank

    @staticmethod
    def _get_last_non_deal_move(last_moves):
        """ Returns last non deal move
        Args:
            last_moves: [pyhanabi.HanabiHistoryItem()]
        Returns:
             last_non_deal_move: pyhanabi.HanabiHistoryItem()
            """
        assert last_moves is not None
        assert isinstance(last_moves, list)

        if len(last_moves) > 0:
            #print('hello', type(last_moves[0]))
            #assert isinstance(last_moves[0], pyhanabi.HanabiHistoryItem)
            pass
        i = 0
        last_move = None
        while True:
            if len(last_moves) <= i:
                break
            if last_moves[i].move().type() != HanabiMoveType.DEAL:
                last_move = last_moves[i]
                break
            i += 1

        return last_move

    def reduce_card_candidate_count(self, last_move):
        """
        When a card has been played or discarded, decrement in self.candidate_counts
        :returns card_candidate_idx: the index of the played/discarded card w.r.t. self.candidate_counts
         """
        card_candidate_idx = self._get_idx_candidate_count(last_move)
        assert card_candidate_idx in [i for i in range(len(self.candidate_counts))]
        assert self.candidate_counts[card_candidate_idx] > 0

        self.candidate_counts[card_candidate_idx] -= 1

        return card_candidate_idx

    def get_abs_agent_idx(self, last_move):
        """ Computes absolute agent position given target_offset and current_player_position,
        i.e. (agent_position + target_offset) % num_players
        Args:
            last_move: pyhanabi.HanabiHistoryItem containing information on the action last taken
        """
        return (last_move.player() + last_move.move().target_offset()) % self.num_players

    def _get_info_reveal_move(self, last_move):
        """ Returns color, rank, and agent_index for a given move of type REVEAL_XYZ
        Args:
            last_move: pyhanabi.HanabiHistoryItem containing information on the action last taken
        Returns:
            color, rank,  target_agent_idx, target_cards_idx: 0-based values,
            color is None or -1 for REVEAL_RANK moves and rank is None or -1 for REVEAL_COLOR moves
        """
        assert last_move.move().type() in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]
        target_cards_idx = last_move.card_info_revealed()
        color = last_move.move().color()
        rank = last_move.move().rank()
        target_agent_idx = self.get_abs_agent_idx(last_move)

        return color, rank, target_agent_idx, target_cards_idx

    def _update_hint_mask(self, last_move):
        """
        Updates the hint_mask given last_move

        Args fetched from self._get_info_reveal_move(last_move):
             color: 0-indexed color corresponding to ["R", "Y", "G", "W", "B"]
             rank: 0-indexed rank
             agent_index: absolute position of agent
             cards_revealed: indices of cards touched by a clue, e.g. [0,2,3]
        ---------------------------------------------------------------------
        e.g. for an initial hintmask
        self.hint_mask = [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
                          [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
                          [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
                          [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]]
        and REVEAL_COLOR: 0 to player 0, e.g.
            color, rank, agent_idx, cards_revealed = 0, -1, 0, [0,1]

        the function would update self.hint_mask to
            [[1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
             [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]

        When REVEAL_COLOR: 1 however, the result will be
             [[0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0.]
              [0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0.]
              [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
        ----------------------------------------------------
        Similarly, when target is player 1, the results will be
            [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
             [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]]
        and
            [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
             [0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
             [0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        for REVEAL_COLOR: 0 and REVEAL_RANK: 1 respectively.
         """
        move_type = last_move.move().type()
        assert move_type in [HanabiMoveType.REVEAL_RANK, HanabiMoveType.REVEAL_COLOR]
        color, rank, target_agent_idx, target_cards_idx = self._get_info_reveal_move(last_move)

        # Slots are rows in self.hint_mask corresponding to the cards touched by a REVEAL_XYZ move
        slots = [target_agent_idx * self.hand_size + slot for slot in target_cards_idx]

        # set all other touched slots to 0, unless they have been hinted already
        mask_untouched = [i for i in slots if i not in self.util_mask_slots_hinted]
        self.hint_mask[mask_untouched] = 0

        # add rows that were hinted
        [self.util_mask_slots_hinted.append(slot) for slot in slots]

        if move_type == HanabiMoveType.REVEAL_COLOR:
            col_idx = color * self.num_ranks
            self.hint_mask[slots, col_idx:col_idx + self.num_ranks] = 1
        elif move_type == HanabiMoveType.REVEAL_RANK:
            cols = [rank + self.num_ranks * c for c in range(self.num_colors)]
            for rank in cols:
                self.hint_mask[slots, rank] = 1
        else:
            raise ValueError

    def _reset_slot_hint_mask(self, last_move):
        """ After a card has been played, its corresponding row in the hint_mask must be reset """
        row = last_move.move().card_index()
        if row in self.util_mask_slots_hinted:  # todo check if this notably decreases performance
            self.util_mask_slots_hinted.remove(row)
        self.hint_mask[row, :] = 1
        self.hint_mask[:, self.candidate_counts == 0] = 0  # set to 0 for impossible cards (all copies played)

    # LVL 1
    def update_candidates_and_hint_mask(self, last_moves):
        """ Update public card knowledge given last move, More specifically
         For PLAY and DISCARD moves:
            - reduce card candidate count by 1
            - if last copy was played, set corresponding self.hint_mask columns to 0
            - reset hint_mask for corresponding card slot
         For REVEAL_COLOR and REVEAL_RANK moves
            - update self.hint_mask
         For DEAL moves:
            - skip
        Does nothing if there are no last_moves.
         """

        if last_moves:  # is False on Initialization, i.e. when last_moves is empty, i.e. when last_moves is False
            last_move = self._get_last_non_deal_move(last_moves)  # pyhanabi.HanabiHistoryItem()
            # PLAY and DISCARD moves
            if last_move.move().type() in [HanabiMoveType.PLAY, HanabiMoveType.DISCARD]:
                # Reduce card candidate count by 1 and if last copy was played, set corresponding hint_mask columns to 0
                card_candidate_idx = self.reduce_card_candidate_count(last_move)
                if self.candidate_counts[card_candidate_idx] == 0:
                    self.hint_mask[:, card_candidate_idx] = 0
                self._reset_slot_hint_mask(last_move)

            # REVEAL_COLOR and REVEAL_RANK moves
            elif last_move.move().type() in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]:
                self._update_hint_mask(last_move)
            else:
                raise ValueError

    def compute_B0_belief(self):
        """ Computes initial public belief that only depends on card counts and hint mask. C.f. eq(12)
        Can be used for baseline agents as well as re-marginalization initialization

        e.g. for 2 players and 2 colors (5 ranks) at start of the game
        initial belief B0 [[3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 0 of player 0 --> B(f[0])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 1 of player 0 --> B(f[1])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 0 of player 1 --> B(f[2])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]] # corresponds to card 1 of player 1 --> B(f[3])

        """
        B0 = self.candidate_counts * self.hint_mask
        B0_normalized = B0 / np.sum(B0, axis=1, keepdims=True)
        # return B0  # broadcasts self.candidate_counts to rows of self.hint_mask
        return B0_normalized  # broadcasts self.candidate_counts to rows of self.hint_mask

    def compute_belief_at_convergence(self, B_0, k=5):
        """ Computes re-marginalized V1 beliefs C.f. eq(13) """
        # Re-marginalize and save to self.public-belief
        assert isinstance(B_0, np.ndarray)
        B_k = np.copy(B_0)
        B_kplus1 = np.zeros((self.num_players*self.hand_size, self.num_colors*self.num_ranks + 1))
        num_slots = B_k.shape[0]
        for _ in range(k):  # k << 10 suffices for until I improved implementation
            # iterate public_belief rows which is 20 iterations at most
            # for i, slot in enumerate(belief_v1):
            for i in range(num_slots):  # todo remove this loop
                B_kplus1[i] = self.candidate_counts - np.sum(
                    B_k[(np.arange(self.num_players * self.hand_size) != i)],  # pick all other rows
                    axis=0)  # sum across other slots
            B_k = B_kplus1 / np.sum(B_kplus1, axis=1, keepdims=True) * self.hint_mask

        # print(re_marginalized)
        return B_k

    def _remove_inconsistent_samples(self, samples, num_consistent_samples_to_return=3000):
        """
                Samples are taken from the posterior belief, B0 or V1, over the private features
                E.g.
                samples = [[5. 7. 8. ... 0. 5. 0.]  # 3000 samples for first card of first player
                           [0. 4. 6. ... 7. 0. 8.]  # 3000 samples for second card of first player
                           [2. 0. 8. ... 2. 2. 5.]
                           [3. 7. 8. ... 1. 5. 1.]]  # 3000 samples for last card of last player

                where the entries are card_numbers between 0 and (colors * ranks) and rows correspond to slots.

                A sample (column) is called consistent, iff the following conditions hold:
                i) The cards sampled are available according to self.candidate_counts  (not all copies already played)
                ii) The cards sampled are possible according to self.hint_mask (card is possible given history of hints)
                """
        # store indices of consistent sampled by their column index
        consistent_samples = list()
        idx = 0
        # currently this loop takes ~70 to 160 ms to run for 20k samples. todo remove this loop
        for sample in samples.T:  # transposing is fastest, as it just changes the np.strides
            sample_card_counts = np.bincount(sample, minlength=len(self.candidate_counts))
            reduced_counts = self.candidate_counts - sample_card_counts
            # i) check sample consistent with candidate count
            if not reduced_counts[reduced_counts < 0].size:
                pass
            else:
                continue
            # ii) check sample consistent with hint_mask
            for shape, val in np.ndenumerate(sample):
                if self.hint_mask[shape[0], val] == 0:
                    continue
            # if this code is reached, the sample is consistent
            consistent_samples.append(idx)
            if len(consistent_samples) == num_consistent_samples_to_return:
                break
            idx += 1
        return samples[:, consistent_samples]

    def sample_consistent_private_features_from_public_belief(self, num_samples=3000, distr_priv_features='V1'):
        """
        In order to compute the bayesian beliefs (c.f. eq. 14, eq. 15), we need to have likelihood samples of
        the private features. These are computed using samples from the public belief (c.f. eq. 8).
        This method returns n samples of private features, that are ensured to be consistent with the deck.
        """
        # We provide the option of sampling private features from either B0 or V1 by setting distr_priv_features
        # sample from the public belief (either self.B0) or (self.V1)
        assert distr_priv_features in ['B0', 'V1']
        belief = getattr(self, distr_priv_features)  # either self.V1 or self.B0
        sampled = np.zeros(shape=(self.num_players * self.hand_size, num_samples), dtype=np.int)

        # compute card indices
        x_i = np.array([card_index for card_index in range(self.num_ranks * self.num_colors)])
        # and their probabilities per slot per player
        px = belief / np.sum(belief, axis=1, keepdims=True)  # todo remove last 1
        # sample hands from the resulting distribution
        for (i_row, _), row in np.ndenumerate(belief):  # todo remove this loop, i.e. sample multivariate
            sampled[i_row, ] = rv_discrete(values=(x_i, px[i_row, :-1])).rvs(size=num_samples)

        # For now we just compute a fixed number of samples and take the first 3000 consistent ones
        return self._remove_inconsistent_samples(sampled, num_consistent_samples_to_return=3000)

    def compute_likelihood_private_features(self, last_moves, seed=123):
        """ Given an observed action, it computes its probability using the public policy.
         This action in turn determines the likelihood of
        """
        assert last_moves  # raise Error when last_moves is empty, because then there is no action yet
        # Otherwise, generate 3000 consistent samples

        samples = self.sample_consistent_private_features_from_public_belief(num_samples=3000)

        self.debug_utils_counter += 1
        # and get the last action
        last_action = self._get_last_non_deal_move(last_moves)
        suitable_private_features = list()
        private_features_that_lead_to_different_action = list()

        for sample in samples.T:  # todo remove this loop (Cython || C++)
            if not self.last_time_step is None:
                if not self.public_policy is None:
                    cnum = 0
                    sampled_obs = self.last_time_step.observation['state']
                    for card in sample:
                        sampled_obs[cnum*self.bits_per_card:cnum*self.bits_per_card+self.bits_per_card] = self.abs_cnum_to_one_hot(card)
                        cnum += 1
                    obs = {'state': self.last_time_step.observation['state'],  # sampled_obs
                           'mask': self.last_time_step.observation['mask']}

                    # replace 0s until self.len_own_hand_vectorized with encoded sample
                    step_type = self.last_time_step.step_type
                    reward = self.last_time_step.reward
                    discount = self.last_time_step.discount
                    timestep = TimeStep(step_type, reward, discount, obs)
                    # compute forward passes using ts
                    # assume that action has been sampled using the public seed
                    # print(f'timestep after replacement = {timestep}')
                    # todo maybe enable eager mode for tf
                    policy_step = self.public_policy.action(self.last_time_step)
                    action = policy_step.action
                    log_prob = getattr(policy_step.info, CommonFields.LOG_PROBABILITY)  # maybe xD
                    # print(f'logprob is {log_prob}')
                    if action == last_action:
                        # can compute the likelihood on the fly, without lists
                        suitable_private_features.append(log_prob)
                    # todo compute the actual values for the likelihood matrix, which is easy once the encoding is given

        return .1

    def abs_cnum_to_one_hot(self, cnum):
        one_hot_enc = np.zeros((self.bits_per_card, ))
        one_hot_enc[cnum] = 1
        return one_hot_enc

    # LVL 2
    def compute_bayesian_belief(self, last_moves, k=1):
        """ Computes re-marginalized Bayesian beliefs. C.f. eq(14) and eq(15)"""
        # Compute basic bayesian belief
        ll = self.compute_likelihood_private_features(last_moves)  # uses self.last_time_step
        BB_k = self.B_0 * ll  # elementwise
        BB_kplus1 = np.zeros((self.num_players*self.hand_size, self.num_colors*self.num_ranks + 1))

        for _ in range(k):  # k << 10 suffices
            # iterate public_belief rows which is 20 iterations at most
            for i in range(self.num_slots):
                BB_kplus1[i] = self.candidate_counts - np.sum(
                    BB_k[np.arange(self.num_players * self.hand_size) != i],axis=0) * self.hint_mask[i,:] * ll
            BB_k = BB_kplus1 / np.sum(BB_kplus1, axis=1, keepdims=True)

        return BB_k

    @staticmethod
    def get_last_moves_from_obs(observations):
        """ observations contains all the hands of all the players """

        current_player = observations['current_player']
        obs = observations['player_observations'][current_player]
        return obs['pyhanabi'].last_moves()  # list of pyhanabi.HanabiHistoryItem from most recent to oldest

    def uniform_likelihood(self):
        """ Returns uniform probabilities for likelihood of private features.
        These are used to initialize the bayesian belief on the first turn"""
        unnormalized = np.ones((self.num_slots, self.num_cards))
        return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)

    def get_vectorized(self):
        """ Returns vectorized public state, i.e. V2 and public features """

    # LVL 2
    def get_observed_hands_vectorized(self, observations):
        """ get the portion of the vectorized hle observation that contains the observed hands
         It has length equal to self.len_private_observation

         Luckily for us, the observed hands are always encoded first, so we just have to get the first
         self.len_private_observations bits from
         """
        cur_pid = observations['current_player']
        return observations['player_observations'][cur_pid]['vectorized'][:self.len_private_observation]

    def compute_public_state(self, observations, k=1):
        """ Computes public belief state s_bad shared by all agents"""
        if self.deck_is_empty:
            self.hint_mask[observations['current_player']:, -1] = 1
        last_moves = self.get_last_moves_from_obs(observations)  # may be empty
        self.update_candidates_and_hint_mask(last_moves)
        self.B_0 = self.compute_B0_belief()
        if not last_moves:
            self.V1 = self.B_0
            self.BB = self.B_0 * self.uniform_likelihood()  # on first turn, there was no action
        else:
            self.V1 = self.compute_belief_at_convergence(self.B_0)
            self.BB = self.compute_bayesian_belief(last_moves)
        self.V2 = (1 - self.alpha) * self.BB + self.alpha * self.V1
        self.public_features = np.concatenate(  # Card-counts and hint-mask
            (self.candidate_counts, self.hint_mask.flatten())
        )

        s_bad = np.append(self.public_features, self.V2.flatten())
        # print(f'len inside compute public state = {len(self.public_features), len(self.V2.flatten())}')
        return {'B0': self.B_0,
                'V1': self.V1,
                'BB': self.BB,
                'V2': self.V2,
                'f_pub': self.public_features,
                'vectorized': s_bad}

    # LVL 0
    def reset(self, rewards_config={}):
        """
        Returns initial HLE observation + initial public belief state, i.e.
        calls self.env.reset() and augments the resulting observation
        """
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.candidate_counts = np.append(self.candidate_counts, 0)  # 1 if card is missing in last turn
        self.hint_mask = np.ones((self.num_players * self.hand_size, self.num_colors * self.num_ranks + 1))
        self.hint_mask[:, -1] = 0
        self.util_mask_slots_hinted = list()
        self.public_features = np.concatenate((self.candidate_counts, self.hint_mask.flatten()))
        self.deck_is_empty = False
        self.B_0 = None
        self.BB = None
        self.V1 = None
        self.V2 = None
        self._episode_ended = False
        self.last_time_step = None
        self.debug_utils_counter = 0
        obs = self.env.reset(rewards_config)

        return self._make_observation_all_players(obs)

    def step(self, action):
        """ Returns observation + updated public belief state """
        observations, reward, done, info = self.env.step(action)

        # Determine if it is the final round
        current_player = observations['current_player']
        obs_current_player = observations['player_observations'][current_player]
        deck_size = obs_current_player['deck_size']
        self.deck_is_empty = True if deck_size <= 0 else False

        # augment total state observation and agents observations
        augmented_observations = self._make_observation_all_players(observations)

        return augmented_observations, reward, done, info

    def vectorized_observation_shape(self):
        """ Returns the shape of a vectorized augmented observation as seen by an agent
        Equals observations_spec in case of wrapping it with tf agents
        """
        len_total = self.env.vectorized_observation_shape()[0]
        if self.use_beliefs:
            len_total += self.len_s_bad
        return (len_total, )  # currently 214

    # LVL 1
    def _make_observation_all_players(self, hle_observations):
        """ Takes HLE observations and augments them by public state. """
        if self.use_beliefs:
            # pid = hle_observations['current_player']
            public_belief_state = self.compute_public_state(hle_observations)
            s_bad_vectorized = public_belief_state['vectorized']  # self.public_features, self.V2.flatten()
            # len(self.public_features) ==
            # for each agent, the observations consists of:
            # hle_observation + public belief state + private features
            # the private features are contained in the hle_observations however
            # so we only add zeros our own hand to the public belief state and append to hle_observation


            for player_dict in hle_observations['player_observations']:
                """ 
                Appends to each players vectorized HLE observation, the public state s_bad={f_pub, B}.
                The private features need not explicitly be included, because they are present in the HLE observationa
                already. When reconstructing them for likelihood sampling, we just need to compute the bit versions of 
                the sampled cards. 
                """

                # append vectorized s_bad to hle observation
                #print(f'len hle obs inside make_obs_all_players = {len( player_dict["vectorized"])}')
                #print(f'len(s_bad_vectorized) = {len(s_bad_vectorized)}')
                #print(f's_bad_vectorized = {s_bad_vectorized}')
                player_dict['vectorized'] = np.append(s_bad_vectorized, player_dict['vectorized'])


            hle_observations['s_bad'] = public_belief_state
        return hle_observations

