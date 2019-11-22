import numpy as np

from custom_environment.utils import get_cards_touched_by_hint, card_is_last_copy, get_card_played_or_discarded
from custom_environment.utils import REVEAL_RANK, REVEAL_COLOR, PLAY, DISCARD, COLOR_CHAR


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
        # Load custom reward params from train_eval script config, default to .2 for each param
        attrs = ['_custom_reward', '_penalty_last_hint_token_used']
        default_attr = .2
        for attr in attrs:
            if attr in self.config:
                setattr(self, attr, self.config[attr])
            else:
                setattr(self, attr, default_attr)

        self.history_size = history_size
        self.history = list()  # stores last hints [may have some (PLAY or DISCARD) actions in between]

    def reset(self, config=None):
        """ config will contain the reward table for PBT """

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

    def maybe_change_hint_reward(self, action, state):

        assert action.type() in [REVEAL_COLOR, REVEAL_RANK]

        reward = 0

        # observation info
        obs_cur_player = state.observation(state.cur_player())
        observed_cards_cur_player = obs_cur_player.observed_hands()

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

        return reward

    def hamming_distance(self, action, vectorized_new, vectorized_old, len_vectorized_obs, per_card=False):

        hamming_distance = 0

        # compute Hamming distance between last two given hints
        last_vectorized_obs = np.array(vectorized_old)
        new_vectorized_obs = np.array(vectorized_new)

        bits_per_card = self.num_colors * self.num_ranks + self.num_ranks + self.num_colors  # bits are not added consecutively
        num_bits_per_hand = bits_per_card * self.hand_size

        start = len_vectorized_obs - self.num_players * num_bits_per_hand
        end = start + num_bits_per_hand

        if per_card:
            dist_per_card = [0 for i in range(self.hand_size)]
            for i in range(self.hand_size):
                end_card = start + self.num_colors * self.num_ranks
                dist_per_card[i] = np.count_nonzero(
                    last_vectorized_obs[start:end_card] != new_vectorized_obs[start:end_card]) / (
                                               self.num_colors * self.num_ranks)
                start += end_card
            hamming_distance = dist_per_card
        else:
            hamming_distance = np.count_nonzero(last_vectorized_obs[start:end] != new_vectorized_obs[start:end]) / (
                        self.num_colors * self.num_ranks * self.hand_size)

        # hamming_distance /= num_bits_per_hand

        # print(action, last_vectorized_obs[start:end], new_vectorized_obs[start:end])
        # print(start, end)
        # print(hamming_distance)
        self.update_history(action, vectorized_new)  # used for next hamming distance

        return hamming_distance

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

        if card_is_last_copy(card_discarded, state.discard_pile()):
            fireworks = state.fireworks()
            # dont punish when the card is already played on the fireworks
            if fireworks[card_discarded.color()] > card_discarded.rank():
                return None
            reward = -2 * float(2 / (card_discarded.rank() + 1))

        return reward

    @staticmethod
    def maybe_apply_weight(reward, weight):

        """
         weight = hamming_distance (currently)
         may either be a list, if hamming distance was computed per card, or a float """

        # in case the hint was no 'play'-hint or 'save'-hint, the reward will still be 0
        if reward != 0:
            reward *= np.sqrt(np.sqrt(weight))  #
        else:
            reward = 0.01 * np.sqrt(np.sqrt(weight))

        return reward
