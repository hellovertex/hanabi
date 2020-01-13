import numpy as np
from collections import defaultdict
from custom_environment.utils import get_cards_touched_by_hint, card_is_last_copy, get_card_played_or_discarded
from custom_environment.utils import REVEAL_RANK, REVEAL_COLOR, PLAY, DISCARD, COLOR_CHAR


class RewardMetrics(object):
    def __init__(self, extended_game_config, rewards_config = {}, history_size = 2):
        # Game-type related stats
        self.config = extended_game_config
        self.num_players = self.config['players']
        self.num_ranks = self.config['ranks']
        self.num_colors = self.config['colors']
        self.hand_size = self.config['hand_size']
        self.total_cards_in_deck = np.sum(np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors))

        self.rewards_config = defaultdict(lambda: 0, rewards_config) # unspecified weights will be 0
        
    def reset(self, rewards_config = {}):
        """ config will contain the reward table for PBT """
        for key in rewards_config:
            self.rewards_config[key] = rewards_config[key]


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
                reward += self.rewards_config['hint_playable']
                is_playable = True
                break  # todo: potentially increase the reward, if more than one hinted card is playable

        # if one hinted card is last copy, set reward to CUSTOM_REWARD
        for card in cards_touched:
            if card_is_last_copy(card, state.discard_pile()):
                if is_playable:
                    reward += self.rewards_config['hint_last_copy']
                else:
                    reward = 0.1 * self.rewards_config['hint_last_copy']
                break  # todo: potentially increase the reward, if more than one hinted card is last copy
        if self.rewards_config['hint_penalty'] is not None:
            hint_penalty = state.information_tokens() - self.rewards_config['hint_penalty']
        else:
            hint_penalty = 1
            
        reward *= hint_penalty
        
        return reward
    def hamming_distance(self, action, vectorized_new, vectorized_old, len_vectorized_obs, per_card=False):

        hamming_distance = 0

        # compute Hamming distance between last two given hints
        last_vectorized_obs = np.array(vectorized_old)
        new_vectorized_obs = np.array(vectorized_new)

        bits_per_card = self.num_colors * self.num_ranks + self.num_ranks + self.num_colors  # bits are not added 
                                                                                             # consecutively
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
       

        return hamming_distance

    def maybe_change_play_reward(self, action, state):

        """ Changes reward for PLAY moves """
        assert action.type() == PLAY

        # get pyhnabi.HanabiCard object for played card
        card_played = get_card_played_or_discarded(action, state.player_hands()[state.cur_player()])
        fireworks = state.fireworks()
        
        if card_played.rank() == fireworks[card_played.color()]:
            reward = self.rewards_config['play' + str(card_played.rank())]
        else:
            reward = self.rewards_config['loose_life']
            #reward = (state.score() + 1) * self.rewards_config['loose_life']
        return reward

    def maybe_change_discard_reward(self, action, state):

        """ Changes reward for DISCARD moves """
        assert action.type() == DISCARD

        card_discarded = get_card_played_or_discarded(action, state.player_hands()[state.cur_player()])


        fireworks = state.fireworks()
        
        if fireworks[card_discarded.color()] > card_discarded.rank():
            reward = self.rewards_config['discard_extra']
        elif card_discarded.rank()==4:
            reward=-1 # TODO: DELETE
        elif card_is_last_copy(card_discarded, state.discard_pile()):
            reward = self.rewards_config['discard_last_copy']*float(1 / (card_discarded.rank() + 1))
        else:
            reward=-float(1/ (card_discarded.rank() + 1)) # TODO: DELTE
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
