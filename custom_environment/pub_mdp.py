import numpy as np

from hanabi_learning_environment import pyhanabi as pyhanabi
from hanabi_learning_environment.pyhanabi import color_char_to_idx
from hanabi_learning_environment.pyhanabi import HanabiMoveType
MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]
COLOR_CHAR = ["R", "Y", "G", "W", "B"]  # consistent with hanabi_lib/util.cc


class PubMDP(object):
    """ Sits on top of the hanabi-learning-environment and is responsible for computation of beliefs (-updates) that
    are used to augment an agents observation to match the state definition of Pub-MDPs
    c.f. 'Bayesian action decoder for deep multi-agent reinforcement learning' (Foerster et al.)
    (https://arxiv.org/abs/1811.01458)"""

    def __init__(self, game_config):
        self.num_colors = game_config['colors']
        self.num_ranks = game_config['ranks']
        self.num_players = game_config['players']
        self.hand_size = game_config['hand_size']
        # 0-1 vector of length (colors * ranks) containing public card counts
        self.candidate_counts = np.tile([3,2,2,2,1][:self.num_ranks], self.num_colors)
        # public binary matrix, slot equals 1, if player could be holding card at given slot
        # i.e. hint_mask[i,j] == 1 <==> (i % hand_size)th card of (i\hand_size)th player equals card j
        self.hint_mask = np.ones((self.num_players * self.hand_size, self.num_colors * self.num_ranks))

        # these will be initialized on env.reset() call
        self.private_features = None  # 0-1 vector of length (num_players * hand_size) * (colors * ranks)
        self.public_features = np.concatenate(  # Card-counts and hint-mask
            (self.candidate_counts, self.hint_mask.flatten())
        )

    def reset(self):
        self.candidate_counts = np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors)
        self.hint_mask = np.ones((self.num_players * self.hand_size, self.num_colors * self.num_ranks))
        self.private_features = None
        self.public_features = np.concatenate((self.candidate_counts, self.hint_mask.flatten()))

    @staticmethod
    def _get_last_non_deal_move(last_moves):
        """ Returns last non deal move
        Args:
            last_moves: [pyhanabi.HanabiHistoryItem()]
        Returns:
             last_non_deal_move: pyhanabi.HanabiMove()
            """
        assert last_moves is not None
        assert isinstance(last_moves, list)
        assert isinstance(last_moves[0], pyhanabi.HanabiMove)
        i = 0
        last_move = None
        while True:
            if len(last_moves) <= i:
                break
            if last_moves[i].type() != HanabiMoveType.DEAL:
                last_move = last_moves[i]
                break
            i += 1
        return last_move

    def update_candidates_and_hint_mask(self, obs):
        """ Update public card knowledge according to change induced by last move """
        assert isinstance(obs, dict)
        assert 'pyhanabi' in obs
        last_moves = obs['pyhanabi'].last_moves()  # list of pyhanabi.HanabiHistoryItem from most recent to oldest
        if last_moves is None:
            pass
        else:
            last_move = self._get_last_non_deal_move(last_moves)  # pyhanabi.HanabiMove()
            if last_move.type == HanabiMoveType.PLAY:
                # reduce card candidate count by 1
                # if last copy was played, set corresponding hint_mask slots to 0
                # todo
                pass
            elif last_move.type == HanabiMoveType.DISCARD:
                # reduce card candidate count by 1
                # if last copy was played, set corresponding hint_mask slots to 0
                # todo
                pass
            elif last_move.type in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]:
                # update hint_mask
                # todo
                pass
            else:
                raise ValueError

    def compute_B0_belief(self):
        """ Computes initial public belief that only depends on card counts and hint mask. C.f. eq(12)
        Can be used for baseline agents as well as re-marginalization init"""
        return (self.candidate_counts * self.hint_mask).flatten()

    def compute_belief_at_convergence(self, k=1):
        """ Computes re-marginalized V1 beliefs C.f. eq(13) """
        # Compute basic belief B0
        belief_b0 = self.compute_B0_belief()

        # Re-marginalize and save to self.public-belief
        def _loop_re_marginalization(belief_v1, k):
            """ Returns public believe at convergence C.f eq(10), eq(13) """
            re_marginalized = belief_v1
            for _ in range(k):  # k << 10 suffices
                # iterate public_belief rows which is 20 iterations at most
                for slot in belief_v1:
                    re_marginalized[slot] = self.candidate_counts - np.sum(
                        belief_v1[np.arange(self.num_players * self.hand_size) != slot],  # pick all other rows
                        axis=0)  # sum across other slots
                belief_v1 = re_marginalized
            return belief_v1

        return _loop_re_marginalization(belief_b0, k)

        # return _loop_re_marginalization(k)

    def compute_bayesian_belief(self, prob_last_action, k=1):
        """ Computes re-marginalized Bayesian beliefs. C.f. eq(14) and eq(15)"""
        # Compute basic bayesian belief
        bayesian_belief_b0 = self.compute_B0_belief() * prob_last_action  # todo: replace prob_last_action with vector

        # Re-marginalize and save to self.public-belief
        def _loop_re_marginalization(bayesian_belief, k):
            """ Returns public believe at convergence C.f eq(10), eq(13) """
            re_marginalized = bayesian_belief
            for _ in range(k):  # k << 10 suffices
                # iterate public_belief rows which is 20 iterations at most
                for slot in bayesian_belief:
                    re_marginalized[slot] = self.candidate_counts - np.sum(
                        bayesian_belief[np.arange(self.num_players * self.hand_size) != slot],
                        # pick all other rows
                        axis=0) * prob_last_action  # todo: replace prob_last_action with vector
                bayesian_belief = re_marginalized
            return bayesian_belief

        return _loop_re_marginalization(bayesian_belief_b0, k)

    def augment_observation(self, obs, prob_last_action=None, alpha=.01, k=1):
        """ Adds public belief state s_bad to the vectorized observation obs and returns flattened network input
        s_bad  = {public_belief, public_features}
        Where
            public_features = obs  [can be changed to more sophisticated observation later]
        and
            public_belief = (1-alpha)BB + alpha(V1), c.f. eq(12)-eq(15)
        Args:
            obs: rl_env.step(a)['player_observations']['current_player']
            prob_last_action: probability of the action taken by the last agent,
            used to compute likelihood of private features. Will be set to 1\num_actions on init
            alpha: smoothness of interpolation between V1 and BB
            k: iteration counter for re-marginalization (c.f. eq(10))
        Returns:
            obs_pub_mdp: vectorized observation including public beliefs and features
        """
        self.update_candidates_and_hint_mask(obs)  # todo: do we update card_counts including hand_cards?
        V1 = self.compute_belief_at_convergence()
        BB = self.compute_bayesian_belief(prob_last_action)
        V2 = (1-alpha) * BB + alpha * V1  # todo check if is vector of correct shape
        obs_pub_mdp = np.concatenate(obs['vectorized_observation'], V2)

        return obs_pub_mdp
