import numpy as np
from custom_environment.utils import abs_position_player_target, color_char_to_idx, get_cards_touched_by_hint
from custom_environment.utils import REVEAL_RANK, REVEAL_COLOR, PLAY, DISCARD


class ObservationAugmenter(object):
    """
    Computes values for extra dimensions added to default state space, using a given strategy.
    These extra values lead to augmented observations.
    The augmented observations are later used e.g. as neural network input.
    """

    def __init__(self, config, history_size=20):
        # each xtra_dim corresponds to one card, i.e. num_extra_state_dims = num_players * hand_size
        self.num_extra_state_dims = config['num_players'] * config['hand_size']
        self.xtra_dims = np.zeros((self.num_extra_state_dims,), dtype=int)

        # history_size determines the number of turns, after which the value of xtra_dim is forgotten
        self.history_size = history_size
        # keep track of 'age' of each value of xtra_dims. Values will be reset, when age > self.history_size
        self.observation_ages = np.zeros(self.xtra_dims.shape, dtype=int)
        # game config
        self.num_players = config['num_players']
        self.hand_size = config['hand_size']
        self.num_colors = config['num_colors']
        self.num_ranks = config['num_ranks']

    def reset(self):
        # see __init__
        self.xtra_dims = np.zeros((self.num_extra_state_dims,), dtype=int)
        self.observation_ages = np.zeros(self.xtra_dims.shape, dtype=int)

    def _indices_of_xtra_dims_affected_by_action(self, action, player_hands, cur_player):
        """
        Computes the indices of extra dimensions in augmented state, that are affected by action.
        Args:
            action: pyhanabi.HanabiMove object, containing the current action
            player_hands: list of (list of pyhanabi.HanabiCard objects) constitung hands of each player in the game
            cur_player: absolute position (pid) of player that computed the action
        Returns:
            idxs_dims: list of integers, containing dimensions indices
        """
        idxs_dim = list()

        # player ID (pid), i.e. absolute position on table, of the target of the action
        abs_pid = abs_position_player_target(action, cur_player, self.num_players)

        # indices of extra dimensions affected by action
        if action.type() in [PLAY, DISCARD]:
            # for play and discard, its always a single index
            idxs_dim = [(abs_pid * self.hand_size) + action.card_index()]

        elif action.type() in [REVEAL_RANK, REVEAL_COLOR]:
            idxs_cards_touched = get_cards_touched_by_hint(hint=action,
                                                           target_hand=player_hands[abs_pid], return_indices=True)
            # index of extra dimensions affected by hint
            idxs_dim = [(abs_pid * self.hand_size) + idx for idx in idxs_cards_touched]

        return idxs_dim

    def _maybe_reset_xtra_dims_given_history_size(self):
        """ Increments the age counter for each xtra_dim and resets (forgets its value) if necessary.
        A value will be reset, if its age_counter is larger than self.history_size.
        """
        # we keep track of the age of the values of the xtra_dims, so increment them here.
        # each index of self.observation_ages corresponds to one index of self.xtra_dims
        self.observation_ages += 1
        # reset those observation dimensions, which have a value older than self.history_size
        self.xtra_dims[self.observation_ages > self.history_size] = 0

    def _apply_strategy(self, target_dims, action):
        """
        Sets values for target_dimensions, according to strategy
        Args:
            target_dims: list of indices corresponding to extra dimensions affected by action
            action: pyhanabi.HanabiMove object containing current action
        """
        # todo in case the strategy changes, we can start from here and will probably not have to change much elsewhere
        xdim_value = None
        # On PLAY/DISCARD moves, we set value of xtra_dim corresponding to the played card equal to 0
        if action.type() in [PLAY, DISCARD]:
            xdim_value = 0
        # On HINT moves, we set the value of xtra_dims corresponding to touched cards according to given strategy
        elif action.type() == REVEAL_COLOR:
            # apply hint encoding
            xdim_value = 1 + action.color()
        elif action.type() == REVEAL_RANK:
            xdim_value = self.num_colors + action.rank()
        else:
            raise ValueError

        self.xtra_dims[target_dims] = xdim_value
        return self.xtra_dims

    @staticmethod
    def _replace_vectorized_inside_observation_by_augmented(observation, augmentation):
        """
        Replaces the observation of the next player inside the observation dictionary by its augmented version.
        The other players observations will be discarded at training time anyway, so dont bother augmenting them as well

        Note: This is the observation, AFTER applying the action, so next_pid is current player in observation

        Args:
            observation: a dict, containing observations for all players, as returned by a call to
                         HanabiEnv._make_observation_all_players
            augmentation: 1D-numpy array containing values for augmented dimensions. Usually its self.xtra_dims
        Returns:
            observation with augmented vectorized_observation for next player
        """
        assert isinstance(observation, dict)
        assert augmentation is not None
        # extract only the vectorized observation of the next agent, this one will be augmented
        next_pid = observation['current_player']
        vectorized_observation = observation['player_observations'][next_pid]['vectorized']
        # concate with augmentation
        augmented_vectorized_observation = vectorized_observation + list(augmentation)
        # replace old vectorized observatoin of next player with new augmented version
        observation['player_observations'][next_pid]['vectorized'] = augmented_vectorized_observation

        return observation

    # entry point for HanabiEnv
    def augment_observation(self, observation, player_hands=None, cur_player=None, action=None):
        """
        Augments the observation we got from environment.step(action), by using a given strategy.

        Since action was computed by cur_player(last player), the observation we want to augment,
        is the observation of the player who comes next, because this player uses the augmented observation,
        to compute the next action and so forth...

        The observation of the next player is replaced inside the observation dictionary by its augmented version

        Args:
            observation: a dict, containing observations for all players, as returned by a call to
                         HanabiEnv._make_observation_all_players
            action: pyhanabi.HanabiMove object, containing the current action
            player_hands: list of (list of pyhanabi.HanabiCard objects) constitung hands of each player in the game as
                          seen by cur_player.
            cur_player: index of player that computed action
        Returns:
            augmented_observation: a new version of observations dict,
            where vectorized_observation of next player has been replaced by concatenation of
            - vectorized_observation
            - self.xtra_dims (after calling self._apply_strategy)
        """

        # Compute augmentation of observation
        if action is None:
            # if action is None, the environment has been reset, then just return zeros for the augmented state
            augmentation = np.zeros((self.num_extra_state_dims,), dtype=int)
        else:
            # indices of extra dimensions in augmented state, that are affected by action
            affected_xtra_dims = self._indices_of_xtra_dims_affected_by_action(action, player_hands, cur_player)
            # Forget(i.e. set to 0) values of self.xtra_dims that are too old, according to history_size
            self._maybe_reset_xtra_dims_given_history_size()
            # set new values for affected_xtra_dims, according to strategy
            augmentation = self._apply_strategy(affected_xtra_dims, action)

        # The observation of the next player is replaced inside the observation dictionary by its augmented version
        augmented_observation = self._replace_vectorized_inside_observation_by_augmented(observation, augmentation)

        return augmented_observation

    # ========== OPEN HANDS MODE ==============

    def vectorize_single_hand(self, hand, all_hands):
        """
        Returns binary representation of hand, consistent with encoding scheme of pyhanabi.ObservationEncoder
        Args:
            hand: a list containing the hand cards, (card dictionaries), e.g.
            [{'color': 'W', 'rank': 2},
             {'color': 'Y', 'rank': 4},
             {'color': 'Y', 'rank': 2},
             {'color': 'G', 'rank': 0},
             {'color': 'W', 'rank': 1}]
            all_hands: list of hands, used to determine bits for missing cards in player hands (on last round of game)

        Returns: obs_vec: a binary list of length hand_size * num_colors * num_ranks
        """
        num_cards = 0
        bits_per_card = self.num_colors * self.num_ranks
        obs_vec = [0 for _ in range(self.hand_size * bits_per_card)]
        offset = 0

        for card in hand:
            rank = card["rank"]
            color = color_char_to_idx(card["color"])
            card_index = color * self.num_ranks + rank

            obs_vec[offset + card_index] = 1
            num_cards += 1
            offset += bits_per_card

        assert len(obs_vec) == self.hand_size * self.num_colors * self.num_ranks

        return obs_vec

    def show_cards(self, observation):
        """
        Uses (for simplicity) previous agents observation, to reveal to current agent his own cards
        It does not matter, whether the previous agent actually acted (i.e. the environment has been reset)
        His observation will contain the necessary information anyways.
        Args:
             observation: dict, containing the full observation about the game at the current step.
                          The current_player is the one for which we have to reveal his cards.

        Returns:
            observation: the same dict, with cards revealed (only to acting player)
        """
        assert isinstance(observation, dict)
        hand_to_reveal = None

        # get previous players observation
        for obs in observation['player_observations']:
            # current_player_offset returns the player index of the acting player, relative to observer.
            if obs['current_player_offset'] == self.num_players - 1:
                prev_player_observation = obs
                # get current agents own cards, knowing that he must have offset 1 to prev player
                hand_to_reveal = prev_player_observation['observed_hands'][1]
                break

        # reveal hand (not necessary for training basically)
        current_player = observation['current_player']
        observation['player_observations'][current_player]['observed_hands'][0] = hand_to_reveal

        # refresh vectorized (by simply appending the agents own vectorized hand to the vectorized observation)
        util_hands = observation['player_observations'][current_player]['observed_hands']
        binary_revealed = self.vectorize_single_hand(hand_to_reveal, util_hands)

        observation['player_observations'][current_player]['vectorized'] += binary_revealed

        return observation

