import numpy as np

from custom_environment.utils import get_cards_touched_by_hint, card_is_last_copy, get_card_played_or_discarded

REVEAL_COLOR = 3  # matches HanabiMoveType.REVEAL_COLOR
REVEAL_RANK = 4  # matches HanabiMoveType.REVEAL_RANK
PLAY = 1  # matches HanabiMoveType.REVEAL_RANK
DISCARD = 2  # matches HanabiMoveType.REVEAL_RANK
COPIES_PER_CARD = {'0': 3, '1': 2, '2': 2, '3': 2, '4': 1}


# todo @gin.configurable
class RewardMetrics(object):
    def __init__(self, extended_game_config, history_size=2):
        # Game-type related stats
        self.config = extended_game_config
        self.num_players = self.config['players']
        self.num_ranks = self.config['ranks']
        self.num_colors = self.config['colors']
        self.hand_size = self.config['hand_size']
        self.total_cards_in_deck = np.sum(np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors))

        # Custom reward params
        self._custom_reward = None
        self._penalty_last_hint_token_used = None
        # Load custom reward params from train_eval script.
        # If not started from there, e.g. from testscript, default to .2 for each param
        attrs = ['_custom_reward', '_penalty_last_hint_token_used']
        default_attr = .2
        for attr in attrs:
            if attr in self.config:
                setattr(self, attr, self.config[attr])
            else:
                setattr(self, attr, default_attr)

        self.history_size = history_size
        self.history = list()  # stores last hints [may have some (PLAY or DISCARD) actions in between]

    def reset(self):
        self.history = list()

    @property
    def penalty_last_hint_token_used(self):
        return self._penalty_last_hint_token_used

    @property
    def custom_reward(self):
        return self._custom_reward

    def update_history(self, action, vectorized_obs):
        """
        Inserts tuple (hint move, vectorized_obs) to FIFO queue self.history.

        Args:
            action: a pyhanabi.HanabiMove object
            vectorized_obs: vectorized observation as returned by pyhanabi.ObservationEncoder

        """
        assert action.type() in [REVEAL_COLOR, REVEAL_RANK]
        if len(self.history) < self.history_size:
            self.history.append((action, vectorized_obs))
        else:
            self.history = self.history[1:]  # remove earliest hint
            self.history.append((action, vectorized_obs))

    def compute_hamming_distance(self, vectorized_obs, len_vectorized_obs):
        """
        Returns the hamming distance between
            vectorized observed state when previous hint given
        and
            vectorized observed state when current hint was given
        relative to acting player.

        When self.history is empty, 0 is returned.
        Args:
             vectorized_obs: vectorized observation as returned by pyhanabi.ObservationEncoder
             len_vectorized_obs: length of vectorized observation. Used for normalization
        Returns:
            hamming_distance: an integer value between 0 and 1
         """

        if len(self.history) == 0:
            return 1  # default for first hint
        else:
            last_vectorized_obs = np.array(self.history[-1][1])
            new_vectorized_obs = np.array(vectorized_obs)
            # in order for the hamming distance to not be too large, we normalize with 1/len
            hamming_distance = np.count_nonzero(last_vectorized_obs != new_vectorized_obs)
            # todo: exclude certain bits for comparison, e.g. those not contributing to entropy
            # todo: compute modified hamming_distance
            return hamming_distance / len_vectorized_obs

    def compute_hamming_distance_for_each_card_seperately(self):
        pass

    def maybe_change_hint_reward(self, action, state, observation_encoder):

        assert action.type() in [REVEAL_COLOR, REVEAL_RANK]

        reward = 0

        # observation info
        obs_cur_player = state.observation(state.cur_player())
        observed_cards_cur_player = obs_cur_player.observed_hands()
        vectorized = observation_encoder.encode(obs_cur_player)

        # action info
        target_offset = action.target_offset()
        target_hand = observed_cards_cur_player[target_offset]

        # get hinted cards
        cards_touched = get_cards_touched_by_hint(hint=action, target_hand=target_hand)
        is_playable = False

        # if one hinted card is playable, set reward to CUSTOM_REWARD
        for card in cards_touched:
            if card.rank() == state.fireworks()[card.color()]:
                reward = self.custom_reward
                is_playable = True
                break  # todo: potentially increase the reward, if more than one hinted card is playable

        # if one hinted card is last copy, set reward to CUSTOM_REWARD
        for card in cards_touched:
            if card_is_last_copy(card, state.discard_pile()):
                if is_playable:
                    reward += self.custom_reward
                else:
                    reward = 0.1 * self.custom_reward
                break  # todo: potentially increase the reward, if more than one hinted card is last copy

        # compute penalty for last hint token used
        hint_penalty = state.information_tokens() - self.penalty_last_hint_token_used
        reward *= hint_penalty

        # compute Hamming distance between last two given hints
        hamming_distance = self.compute_hamming_distance(vectorized, observation_encoder.shape()[0])
        self.update_history(action, vectorized)  # used for next hamming distance

        # in case the hint was no 'play'-hint or 'save'-hint, the reward will still be 0
        if reward != 0:
            reward *= np.sqrt(np.sqrt(hamming_distance))
        else:
            reward = 0.01 * np.sqrt(np.sqrt(hamming_distance))

        return reward

    @staticmethod
    def maybe_change_play_reward(action, state):

        """ Changes reward for PLAY moves """
        assert action.type() == PLAY

        # get pyhnabi.HanabiCard object for played card
        card_played = get_card_played_or_discarded(action, state.player_hands()[state.cur_player()])
        fireworks = state.fireworks()

        reward = None
        if card_played.rank() in [2, 3, 4]:
            if card_played.rank() == fireworks[card_played.color()] and fireworks[0] > 0 and fireworks[1] > 0:
                reward = 2 ** card_played.rank()
            if card_played.rank() == fireworks[card_played.color()] and fireworks[0] > 1 and fireworks[1] > 1:
                reward = 5 ** card_played.rank()
            if card_played.rank() == fireworks[card_played.color()] and fireworks[0] > 2 and fireworks[1] > 2:
                reward = 10 ** card_played.rank()

        return reward

    @staticmethod
    def maybe_change_discard_reward(action, state):

        """ Changes reward for DISCARD moves """
        assert action.type() == DISCARD

        reward = None
        card_discarded = get_card_played_or_discarded(action, state.player_hands()[state.cur_player()])

        # todo punish discarding last copies of cards, weighted inversely by their rank
        if card_is_last_copy(card_discarded, state.discard_pile()):
            # dont punish when the card is already played on the fireworks

            reward = -2 * float(2 / (card_discarded.rank() + 1))

        return reward