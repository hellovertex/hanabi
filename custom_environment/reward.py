"""Implements various custom rewards.

A RewardMetrics object is added to a HanabiEnv. The "partial" reward functions are called each .step(),
depending on flags set i.e. which custom rewards are used.

"""
import numpy as np
# import gin.tf

from custom_environment.utils import get_cards_touched_by_hint, card_is_last_copy, get_card_played_or_discarded
from custom_environment.utils import REVEAL_RANK, REVEAL_COLOR, PLAY, DISCARD, COLOR_CHAR
from collections import namedtuple

class RewardMetrics(object):
    """RewardMetrics object providing custom rewards to be provided to an agent from within a HanabiEnvironment.
    """
    def __init__(self, extended_game_config, history_size=2):
        """Creates a RewardMetrics object with given extended_game_config.

                Args:
                  extended_game_config: dict, With parameters for the custom rewards. Takes the following
                    keys and values.
                      - colors: int, Number of colors \in [2,5].
                      - ranks: int, Number of ranks \in [2,5].
                      - players: int, Number of players \in [2,5].
                      - hand_size: int, Hand size \in [4,5].
                      - per_card_reward=True: bool, whether or not to use individual rewards per card #todo: what?
                      - _custom_reward=.2: float, weight for custom reward, seems like not yet used?
                      - _penalty_last_hint_token_used=.2: float, weight for custom reward
                """
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
        attrs = ['per_card_reward', '_custom_reward', '_penalty_last_hint_token_used']
        default_vals = [True, .2, .2]
        for attr, def_val in zip(attrs, default_vals):
            if attr in self.config:
                setattr(self, attr, self.config[attr])
            else:
                setattr(self, attr, def_val)

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
        is_last_copy = False

        # Compute single reward
        # todo: (re-)move this if statement (is it needed anymore?)
        if not self.per_card_reward:
            # if one hinted card is playable, set reward to CUSTOM_REWARD
            for card in cards_touched:
                if card.rank() == state.fireworks()[card.color()]:
                    reward = self.custom_reward
                    is_playable = True
                    break

            # if one hinted card is last copy, set reward to CUSTOM_REWARD
            for card in cards_touched:
                if card_is_last_copy(card, state.discard_pile()):
                    if is_playable:
                        reward += self.custom_reward
                    else:
                        reward = 0.1 * self.custom_reward
                    break
        # Compute reward per card
        else:

            # todo can add more conditions here
            cond = namedtuple('card_condition', ['playable', 'last_copy'])
            # initialize conditions per card
            conds_per_card = [cond(playable=False, last_copy=False) for _ in range(self.hand_size)]
            reward_per_card = [0 for _ in range(self.hand_size)]

            # set reward values
            reward_playable = self.custom_reward
            reward_last_copy = self.custom_reward * .1

            # determine for each card, which conditions are True
            for i, card in enumerate(cards_touched):
                if card.rank() == state.fireworks()[card.color()]:
                    is_playable = True
                if card_is_last_copy(card, state.discard_pile()):
                    is_last_copy = True
                conds_per_card[i] = cond(is_playable, is_last_copy)
            # Compute the reward accordingly
                reward_per_card[i] = conds_per_card[i].playable * reward_playable + \
                                     conds_per_card[i].last_copy * reward_last_copy
                # reset conditions
                is_playable = False
                is_last_copy = False

            reward = np.array(reward_per_card)

        # compute penalty for last hint token used
        hint_penalty = state.information_tokens() - self.penalty_last_hint_token_used
        reward *= hint_penalty

        return reward

    def hamming_distance(self, action, vectorized_new, vectorized_old, len_vectorized_obs):

        hamming_distance = 0

        # compute Hamming distance between last two given hints
        last_vectorized_obs = np.array(vectorized_old)
        new_vectorized_obs = np.array(vectorized_new)

        bits_per_card = self.num_colors * self.num_ranks + self.num_ranks + self.num_colors  # bits are not added consecutively
        num_bits_per_hand = bits_per_card * self.hand_size

        start = len_vectorized_obs - self.num_players * num_bits_per_hand
        end = start + num_bits_per_hand

        if self.per_card_reward:
            dist_per_card = np.array([0. for i in range(self.hand_size)])
            for i in range(self.hand_size):
                end_card = start + self.num_colors * self.num_ranks
                # (ones before - ones after) / ones after
                # normalization_before_commit = self.num_colors * self.num_ranks
                normalization = np.count_nonzero(new_vectorized_obs[start:end_card]) + 1  # add one in case its 0
                difference = np.count_nonzero(
                    last_vectorized_obs[start:end_card] != new_vectorized_obs[start:end_card])
                dist_per_card[i] = difference / normalization
                # print(f'action is {action} difference is {difference}, normalization is {normalization}, dist_per_card[i] is {dist_per_card[i]}, new bits { new_vectorized_obs[start:end_card]}')
                start += end_card + self.num_ranks + self.num_colors
            hamming_distance = dist_per_card
        else:
            # this is approximate, as end may contain some extra bits due to bits_per_card value
            # normalization_before_commit = self.num_colors * self.num_ranks * self.hand_size
            normalization = np.count_nonzero(new_vectorized_obs[start:end])
            hamming_distance = np.count_nonzero(last_vectorized_obs[start:end] != new_vectorized_obs[start:end]) / normalization


        self.update_history(action, vectorized_new)  # used for next hamming distance

        return hamming_distance

    @staticmethod
    def maybe_change_play_reward(action, state):

        """ Changes reward for PLAY moves """
        assert action.type() == PLAY

        # get pyhnabi.HanabiCard object for played card
        card_played = get_card_played_or_discarded(action, state.player_hands()[state.cur_player()])
        fireworks = state.fireworks()


        if card_played.rank() in [0, 1, 2, 3, 4]:
            if card_played.rank() == fireworks[card_played.color()]:
                reward = 2 ** card_played.rank()
            if card_played.rank() != fireworks[card_played.color()]:
                reward= -10 #todo: this is maybe a little much?
        return reward

    @staticmethod
    def maybe_change_discard_reward(action, state):

        """ Changes reward for DISCARD moves """
        assert action.type() == DISCARD

        card_discarded = get_card_played_or_discarded(action, state.player_hands()[state.cur_player()])


        fireworks = state.fireworks()

        if fireworks[card_discarded.color()] > card_discarded.rank():
            reward=0.5
        elif card_discarded.rank()==4:
            reward=-1
        elif card_is_last_copy(card_discarded, state.discard_pile()):
            reward= -20 * float(2 / (card_discarded.rank() + 1))
        else:
            reward=-float(1/ (card_discarded.rank() + 1))
        return reward

    def maybe_apply_weight(self, reward, weight):

        """
         weight = hamming_distance (currently)
         may either be a list, if hamming distance was computed per card, or a float """

        # in case the hint was no 'play'-hint or 'save'-hint, the reward will still be 0
        if isinstance(weight, list) or isinstance(weight, np.ndarray):
            assert (isinstance(reward, list) or isinstance(reward, np.ndarray))
            assert self.per_card_reward is True
            # compute weighting elementwise
            return np.log(np.sum(np.exp(5*np.multiply(weight, reward))))/5 + 0.01*np.sum(weight)

        return reward

# #todo: this should probably be inside class?
# def customize_reward(agent, raw_reward, action):
#     reward = 0
#     prev_player_hands = self.state.player_hands()
#     # needed for hamming distance
#     cur_pid = self.state.cur_player()  # absolute pid
#     next_pid = (cur_pid + 1) % self.game.num_players()  # absolute next_pid
#
#     old_obs_next_player = self.state.observation(next_pid)
#     vectorized_old = self.observation_encoder.encode(old_obs_next_player)
#
#
#     if self.USE_HINT_REWARD and (action.type() in [REVEAL_COLOR, REVEAL_RANK]):
#         hint_reward = self.reward_metrics.maybe_change_hint_reward(action, self.state)
#         reward += hint_reward
#         info["hint_reward"] = hint_reward
#     if self.USE_PLAY_REWARD and (action.type() == PLAY):
#         play_reward = self.reward_metrics.maybe_change_play_reward(action, self.state)
#         reward += play_reward
#         info["play_reward"] = play_reward
#     if self.USE_DISCARD_REWARD and (action.type() == DISCARD):
#         discard_reward = self.reward_metrics.maybe_change_discard_reward(action, self.state)
#         reward += discard_reward + 3 #TODO: discard_reward is not balanced yet,this is only a quick and dirty fix
#     info["discard_reward"] = discard_reward