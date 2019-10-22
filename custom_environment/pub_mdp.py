import numpy as np

from hanabi_learning_environment import pyhanabi as pyhanabi
from hanabi_learning_environment.pyhanabi import color_char_to_idx
from hanabi_learning_environment.pyhanabi import HanabiMoveType
MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]
COLOR_CHAR = ["R", "Y", "G", "W", "B"]  # consistent with hanabi_lib/util.cc


class NNstub(object):
    """ Neural Network stub for the public policy, until we decided on an architecture"""

    @staticmethod
    def feedforward(observation):
        """
        Returns action probabilities given observation as input
        """
        return .1


class PubMDP(object):
    """ Sits on top of the hanabi-learning-environment and is responsible for computation of beliefs (-updates) that
    are used to augment an agents observation to match the state definition of Pub-MDPs
    c.f. 'Bayesian action decoder for deep multi-agent reinforcement learning' (Foerster et al.)
    (https://arxiv.org/abs/1811.01458)"""

    def __init__(self, game_config):
        """
            See definition of augment_observation() at the bottom of this file to understand how this class works
        """
        self.num_colors = game_config['colors']
        self.num_ranks = game_config['ranks']
        self.num_players = game_config['players']
        self.hand_size = game_config['hand_size']
        # C(f)
        # 0-1 vector of length (colors * ranks) containing public card counts
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.candidate_counts = np.append(self.candidate_counts, 0)  # 1 if card is missing in last turn
        # HM
        # public binary matrix, slot equals 1, if player could be holding card at given slot
        # i.e. hint_mask[i,j] == 1 <==> (i % hand_size)th card of (i\hand_size)th player equals card j
        self.hint_mask = np.ones((self.num_players * self.hand_size, self.num_colors * self.num_ranks + 1))
        self.hint_mask[:, -1] = 0
        # utils
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
        self.public_features = np.concatenate(  # Card-counts and hint-mask
            (self.candidate_counts, self.hint_mask.flatten())
        )

        self.public_policy = NNstub()
        self.B0 = None
        self.BB = None
        self.V1 = None
        self.V2 = None

    def reset(self):
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.candidate_counts = np.append(self.candidate_counts, 0)  # 1 if card is missing in last turn
        self.hint_mask = np.ones((self.num_players * self.hand_size, self.num_colors * self.num_ranks + 1))
        self.hint_mask[:, -1] = 0
        self.util_mask_slots_hinted = list()
        self.public_features = np.concatenate((self.candidate_counts, self.hint_mask.flatten()))
        self.deck_is_empty = False

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
    def update_candidates_and_hint_mask(self, obs):
        """ Update public card knowledge according to change induced by last move, More specifically
         For PLAY and DISCARD moves:
            - reduce card candidate count by 1
            - if last copy was played, set corresponding self.hint_mask columns to 0
            - reset hint_mask for corresponding card slot
         For REVEAL_COLOR and REVEAL_RANK moves
            - update self.hint_mask
         For DEAL moves:
            - skip
         """
        assert isinstance(obs, dict)
        assert 'pyhanabi' in obs

        last_moves = obs['pyhanabi'].last_moves()  # list of pyhanabi.HanabiHistoryItem from most recent to oldest

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

        if self.deck_is_empty:
            self.hint_mask[:, -1] = 1

    def compute_B0_belief(self):
        """ Computes initial public belief that only depends on card counts and hint mask. C.f. eq(12)
        Can be used for baseline agents as well as re-marginalization init"""
        """
        e.g. for 2 players and 2 colors (5 ranks) at start of the game
        initial belief B0 [[3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 0 of player 0 --> B(f[0])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 1 of player 0 --> B(f[1])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]  # corresponds to card 0 of player 1 --> B(f[2])
                        [3. 2. 2. 2. 1. 3. 2. 2. 2. 1. 0.]] # corresponds to card 1 of player 1 --> B(f[3])

        """
        return (self.candidate_counts * self.hint_mask)

    def compute_belief_at_convergence(self, k=1):
        """ Computes re-marginalized V1 beliefs C.f. eq(13) """
        # Compute basic belief B0
        belief_b0 = self.compute_B0_belief()
        # todo timeit because it adds quadratic complexity despite low coefficient
        # Re-marginalize and save to self.public-belief

        def _loop_re_marginalization(belief_v1, k):
            """ Returns public believe at convergence C.f eq(10), eq(13) """
            # re_marginalized = np.copy(belief_v1)
            re_marginalized = np.zeros((self.num_players*self.hand_size, self.num_colors*self.num_ranks + 1))
            for _ in range(k):  # k << 10 suffices for until I improved implementation
                # iterate public_belief rows which is 20 iterations at most

                for i, slot in enumerate(belief_v1):
                    # print('initial belief', belief_v1)
                    re_marginalized[i] = self.candidate_counts - np.sum(
                        belief_v1[(np.arange(self.num_players * self.hand_size) != i)],  # pick all other rows
                        axis=0)  # sum across other slots
                    # print(i, re_marginalized[i])
                # belief_v1 = re_marginalized
            # print(re_marginalized)
            return re_marginalized

        return _loop_re_marginalization(belief_b0, k)

        # return _loop_re_marginalization(k)

    def compute_likelihood_private_features(self, obs):
        """ Given an observed action, it computes its probability using the public policy.
         This action in turn determines the likelihood of
         """
        # todo compute likelihood matrix self.L([f[i]) with num_agents x (length num_colors * num_ranks) by:
        # todo 1. in observed cards, replace the ones of previously acting agent by your own (unknown)
        # todo 2. feedforward all possible f_a and store the probability of action that equals last taken
        # todo 3. Since you cannot feedforward all possible f_a you need to sample your f_a from B
        # todo 4. multiply the results of 2. with self.LL to compute the result

        # 0. Define f_a: Get common observed hands of you and acting agent and add placeholder for your hand
        # 1. Sample consistent hands from self.V1 if self.BB is None and from self.BB otherwise for your hand
        # ll = 0
        # Loop
        #   replace placeholders in f_a by sample
        #   feedforward f_a and for each store probability p of the action actually taken
        #   ll += self.LL(f[i]) * a
        # ll \= num_samples
        return .1

    # LVL 2
    def compute_bayesian_belief(self, observation, k=1):
        """ Computes re-marginalized Bayesian beliefs. C.f. eq(14) and eq(15)"""
        # Compute basic bayesian belief
        ll = self.compute_likelihood_private_features(observation)
        bayesian_belief_b0 = self.compute_B0_belief() * ll

        # Re-marginalize and save to self.public-belief
        def _loop_re_marginalization(bayesian_belief, k):
            """ Returns public believe at convergence C.f eq(10), eq(13) """
            re_marginalized = np.copy(bayesian_belief)
            for _ in range(k):  # k << 10 suffices
                # iterate public_belief rows which is 20 iterations at most
                for i, slot in enumerate(bayesian_belief):
                    re_marginalized[i] = self.candidate_counts - np.sum(
                        bayesian_belief[np.arange(self.num_players * self.hand_size) != i],
                        # pick all other rows
                        axis=0) * ll
                bayesian_belief = re_marginalized
            return bayesian_belief

        return _loop_re_marginalization(bayesian_belief_b0, k)

    # LVL 0
    def augment_observation(self, obs, alpha=.01, k=1):
        """
        Adds public belief state 's_bad' to a vectorized observation 'obs' gotten from an environment
        Returns the augmented binary observation vector (ready to use NN input)

        s_bad  = {public_features, public_belief}

        Where
            public_features = obs  [can be changed to more sophisticated observation later]
        and
            public_belief = (1-alpha)BB + alpha(V1), c.f. eq(12)-eq(15)
        Args:
            obs: rl_env.step(a)['player_observations']['current_player']
            used to compute likelihood of private features. Will be set to 1\num_actions on init
            alpha: smoothness of interpolation between V1 and BB
            k: iteration counter for re-marginalization (c.f. eq(10))
        Returns:
            obs_pub_mdp: vectorized observation including public beliefs and features
        """
        assert isinstance(obs, dict)
        assert 'current_player_offset' in obs  # ensure to always have agent_observation, not full observation

        self.deck_is_empty = True if obs['deck_size'] <= 0 else False

        self.update_candidates_and_hint_mask(obs)
        self.V1 = self.compute_belief_at_convergence()
        self.BB = self.compute_bayesian_belief(obs)
        self.V2 = (1-alpha) * self.BB + alpha * self.V1  # todo check if is vector of correct shape
        obs['V1_belief'] = self.V1
        obs['BB'] = self.BB
        obs['V2'] = self.V2
        # print(V1)
        # print(BB)
        # obs['obs_pub_mdp_vectorized'] = np.concatenate(obs['vectorized'], V2)

        return obs


class BADAgent(object):
    def __init__(self, game_config, pub_mdp_params=None):
        self.pub_mdp = PubMDP(game_config)

    def act(self, observation):
        obs = self.pub_mdp.augment_observation(observation)
        action = 'action computed by forwardpass using new obs'
        return action
