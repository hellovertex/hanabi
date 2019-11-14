# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RL environment for Hanabi, using an API similar to OpenAI Gym."""

from __future__ import absolute_import
from __future__ import division

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pyhanabi as pyhanabi
from pyhanabi import color_char_to_idx

# custom imports
import numpy as np

MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]

# -------------------------------------------------------------------------------
# Environment API
# -------------------------------------------------------------------------------

USE_CUSTOM_REWARD = True
CUSTOM_REWARD = .1
PENALTY_LAST_HINT_TOKEN_USED = .5
COPIES_PER_CARD = {'0': 3, '1': 2, '2': 2, '3': 2, '4': 1}
REVEAL_COLOR = 3  # matches HanabiMoveType.REVEAL_COLOR
REVEAL_RANK = 4  # matches HanabiMoveType.REVEAL_RANK
PLAY = 1

class Environment(object):
    """Abtract Environment interface.

    All concrete implementations of an environment should derive from this
    interface and implement the method stubs.
    """

    def reset(self, config):
        """Reset the environment with a new config.

        Signals environment handlers to reset and restart the environment using
        a config dict.

        Args:
          config: dict, specifying the parameters of the environment to be
            generated.

        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to an action taken by an agent.

        Returns:
          observation: dict, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


# @gin.configurable
class StorageRewardMetrics(object):
    def __init__(self, extended_game_config, history_len=2):
        # Game-type related stats
        self.config = extended_game_config
        self.num_players = self.config['players']
        self.num_ranks = self.config['ranks']
        self.num_colors = self.config['colors']
        self.hand_size = self.config['hand_size']
        self.total_cards_in_deck = np.sum(np.tile([3, 2, 2, 2, 1][:self.num_ranks], self.num_colors))

        # Custom Reward params
        self._custom_reward = CUSTOM_REWARD
        self._penalty_last_hint_token_used = PENALTY_LAST_HINT_TOKEN_USED
        """ custom settings may be optionally overwritten by extending game_config """
        if 'custom_reward' in self.config:
            self._custom_reward = self.config['custom_reward']
        if 'penalty_last_hint_token' in self.config:
            self._penalty_last_hint_token_used = self.config['penalty_last_hint_token']

        # Game-metric-utils
        starting_pace = self.total_cards_in_deck - (self.hand_size - 1) * self.num_players
        starting_pace -= 5 * self.num_colors
        min_efficiency_numerator = 5 * self.num_colors
        self.history_len = history_len  # todo remove
        self.history = list()  # stores last hints [may have some (PLAY or DISCARD) actions in between]

        # Game metrics
        self._efficiency = None
        self._num_hints_given = 0

        """
        deck_size = None
        num_players = None
        hand_size = None
        starting_pace = None
        num_cards_played = len(self.state.discard_pile) + sum(fireworks.values())    
        """
        # commpute starting pace

    def reset(self):
        self._num_hints_given = 0
        self.history = list()

    @property
    def penalty_last_hint_token_used(self):
        return self._penalty_last_hint_token_used

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        self._efficiency = value

    @property
    def custom_reward(self):
        return self._custom_reward

    @property
    def num_hints_given(self):
        return self._num_hints_given

    @num_hints_given.setter
    def num_hints_given(self, value):
        self._num_hints_given = value

    def update_history(self, action, vectorized_obs):
        """
        Inserts tuple (hint move, vectorized_obs) to FIFO queue self.history.

        Args:
            action: a pyhanabi.HanabiMove object
            vectorized_obs: vectorized observation as returned by pyhanabi.ObservationEncoder

        """
        assert action.type() in [REVEAL_COLOR, REVEAL_RANK]
        if len(self.history) < self.history_len:
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


class HanabiEnv(Environment):
    """RL interface to a Hanabi environment.

    ```python

    environment = rl_env.make()
    config = { 'players': 5 }
    observation = environment.reset(config)
    while not done:
        # Agent takes action
        action =  ...
        # Environment take a step
        observation, reward, done, info = environment.step(action)
    ```
    """

    def __init__(self, config):
        r"""Creates an environment with the given game configuration.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0).
              - max_life_tokens: int, Number of life tokens (>=1).
              - observation_type: int.
                0: Minimal observation.
                1: First-order common knowledge observation.
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
        """
        assert isinstance(config, dict), "Expected config to be of type dict."
        self.game = pyhanabi.HanabiGame(config)

        self.observation_encoder = pyhanabi.ObservationEncoder(
            self.game, pyhanabi.ObservationEncoderType.CANONICAL)
        self.players = self.game.num_players()

        # in case the game_config did not contain hand_size because it was meant to be defaulted
        config['hand_size'] = self.game.hand_size()
        self.reward_metrics = StorageRewardMetrics(config)

    def reset(self):
        r"""Resets the environment for a new game.

        Returns:
          observation: dict, containing the full observation about the game at the
            current step. *WARNING* This observation contains all the hands of the
            players and should not be passed to the agents.
            An example observation:
            {'current_player': 0,
             'player_observations': [{'current_player': 0,
                                      'current_player_offset': 0,
                                      'deck_size': 40,
                                      'discard_pile': [],
                                      'fireworks': {'B': 0,
                                                    'G': 0,
                                                    'R': 0,
                                                    'W': 0,
                                                    'Y': 0},
                                      'information_tokens': 8,
                                      'legal_moves': [{'action_type': 'PLAY',
                                                       'card_index': 0},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 1},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 2},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 3},
                                                      {'action_type': 'PLAY',
                                                       'card_index': 4},
                                                      {'action_type':
                                                      'REVEAL_COLOR',
                                                       'color': 'R',
                                                       'target_offset': 1},
                                                      {'action_type':
                                                      'REVEAL_COLOR',
                                                       'color': 'G',
                                                       'target_offset': 1},
                                                      {'action_type':
                                                      'REVEAL_COLOR',
                                                       'color': 'B',
                                                       'target_offset': 1},
                                                      {'action_type': 'REVEAL_RANK',
                                                       'rank': 0,
                                                       'target_offset': 1},
                                                      {'action_type': 'REVEAL_RANK',
                                                       'rank': 1,
                                                       'target_offset': 1},
                                                      {'action_type': 'REVEAL_RANK',
                                                       'rank': 2,
                                                       'target_offset': 1}],
                                      'life_tokens': 3,
                                      'observed_hands': [[{'color': None, 'rank':
                                      -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1}],
                                                         [{'color': 'G', 'rank': 2},
                                                          {'color': 'R', 'rank': 0},
                                                          {'color': 'R', 'rank': 1},
                                                          {'color': 'B', 'rank': 0},
                                                          {'color': 'R', 'rank':
                                                          1}]],
                                      'num_players': 2,
                                      'vectorized': [ 0, 0, 1, ... ]},
                                     {'current_player': 0,
                                      'current_player_offset': 1,
                                      'deck_size': 40,
                                      'discard_pile': [],
                                      'fireworks': {'B': 0,
                                                    'G': 0,
                                                    'R': 0,
                                                    'W': 0,
                                                    'Y': 0},
                                      'information_tokens': 8,
                                      'legal_moves': [],
                                      'life_tokens': 3,
                                      'observed_hands': [[{'color': None, 'rank':
                                      -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1},
                                                          {'color': None, 'rank':
                                                          -1}],
                                                         [{'color': 'W', 'rank': 2},
                                                          {'color': 'Y', 'rank': 4},
                                                          {'color': 'Y', 'rank': 2},
                                                          {'color': 'G', 'rank': 0},
                                                          {'color': 'W', 'rank':
                                                          1}]],
                                      'num_players': 2,
                                      'vectorized': [ 0, 0, 1, ... ]}]}
        """
        self.state = self.game.new_initial_state()

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        obs = self._make_observation_all_players()
        obs["current_player"] = self.state.cur_player()
        self.reward_metrics.reset()

        return obs

    def vectorized_observation_shape(self):
        """Returns the shape of the vectorized observation.

        Returns:
          A list of integer dimensions describing the observation shape.
        """
        return self.observation_encoder.shape()

    def num_moves(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        return self.game.max_moves()

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to a legal action taken by an agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }
            Alternatively, action may be an int in range [0, num_moves()).

        Returns:
          observation: dict, containing the full observation about the game at the
            current step. *WARNING* This observation contains all the hands of the
            players and should not be passed to the agents.
            An example observation:
            {'current_player': 0,
             'player_observations': [{'current_player': 0,
                                'current_player_offset': 0,
                                'deck_size': 40,
                                'discard_pile': [],
                                'fireworks': {'B': 0,
                                          'G': 0,
                                          'R': 0,
                                          'W': 0,
                                          'Y': 0},
                                'information_tokens': 8,
                                'legal_moves': [{'action_type': 'PLAY',
                                             'card_index': 0},
                                            {'action_type': 'PLAY',
                                             'card_index': 1},
                                            {'action_type': 'PLAY',
                                             'card_index': 2},
                                            {'action_type': 'PLAY',
                                             'card_index': 3},
                                            {'action_type': 'PLAY',
                                             'card_index': 4},
                                            {'action_type': 'REVEAL_COLOR',
                                             'color': 'R',
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_COLOR',
                                             'color': 'G',
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_COLOR',
                                             'color': 'B',
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_RANK',
                                             'rank': 0,
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_RANK',
                                             'rank': 1,
                                             'target_offset': 1},
                                            {'action_type': 'REVEAL_RANK',
                                             'rank': 2,
                                             'target_offset': 1}],
                                'life_tokens': 3,
                                'observed_hands': [[{'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1}],
                                               [{'color': 'G', 'rank': 2},
                                                {'color': 'R', 'rank': 0},
                                                {'color': 'R', 'rank': 1},
                                                {'color': 'B', 'rank': 0},
                                                {'color': 'R', 'rank': 1}]],
                                'num_players': 2,
                                'vectorized': [ 0, 0, 1, ... ]},
                               {'current_player': 0,
                                'current_player_offset': 1,
                                'deck_size': 40,
                                'discard_pile': [],
                                'fireworks': {'B': 0,
                                          'G': 0,
                                          'R': 0,
                                          'W': 0,
                                          'Y': 0},
                                'information_tokens': 8,
                                'legal_moves': [],
                                'life_tokens': 3,
                                'observed_hands': [[{'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1},
                                                {'color': None, 'rank': -1}],
                                               [{'color': 'W', 'rank': 2},
                                                {'color': 'Y', 'rank': 4},
                                                {'color': 'Y', 'rank': 2},
                                                {'color': 'G', 'rank': 0},
                                                {'color': 'W', 'rank': 1}]],
                                'num_players': 2,
                                'vectorized': [ 0, 0, 1, ... ]}]}
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        if isinstance(action, dict):
            # Convert dict action HanabiMove
            action = self._build_move(action)
        elif isinstance(action, int):
            # Convert int action into a Hanabi move.
            action = self.game.get_move(action)
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(
                action))

        # ################################################ #
        # ----------------- Custom Reward ---------------- #
        # ################################################ #
        reward = None
        if USE_CUSTOM_REWARD:
            fireworks = self.state.fireworks()
            cur_player = self.state.cur_player()

            # For hint moves, change the default reward
            if action.type() in [REVEAL_COLOR, REVEAL_RANK]:
                reward = 0
                # Update reward_metric utils
                self.reward_metrics.num_hints_given += 1  # used to compute efficiency later

                # observation info
                obs_cur_player = self.state.observation(cur_player)
                observed_cards_cur_player = obs_cur_player.observed_hands()
                vectorized = self.observation_encoder.encode(obs_cur_player)

                # action info
                target_offset = action.target_offset()
                target_hand = observed_cards_cur_player[target_offset]
                cards_touched = list()  # list of pyhanabi.HanabiMove objects containing hinted cards

                # get hinted cards
                if action.type() == REVEAL_COLOR:
                    color_hinted = action.color()
                    for card in target_hand:
                        if card.color() == color_hinted:
                            cards_touched.append(card)
                elif action.type() == REVEAL_RANK:
                    rank_hinted = action.rank()
                    for card in target_hand:
                        if card.rank() == rank_hinted:
                            cards_touched.append(card)
                else:
                    raise ValueError

                is_playable = False
                is_last_copy = False
                # if one hinted card is playable, set reward to CUSTOM_REWARD
                for card in cards_touched:
                    if card.rank() == fireworks[card.color()]:
                        reward = self.reward_metrics.custom_reward
                        is_playable = True
                        break

                # if one hinted card is last copy, set reward to CUSTOM_REWARD
                for card in cards_touched:
                    card_copies_total = COPIES_PER_CARD[str(card.rank())]
                    card_copies_discarded = 0
                    for discarded in self.state.discard_pile():
                        if discarded.color() == card.color() and discarded.rank() == card.rank():
                            card_copies_discarded += 1
                    if card_copies_total - card_copies_discarded == 1:
                        if is_playable:
                            reward += self.reward_metrics.custom_reward
                        else:
                            reward = 0.1*self.reward_metrics.custom_reward
                        break

                # compute "efficiency" as factor for reward
                # todo if efficiency is used, how do we initialize it?, Zero division prohibits using it atm

                # compute penalty for last hint token used
                hint_penalty = self.state.information_tokens() - self.reward_metrics.penalty_last_hint_token_used
                reward *= hint_penalty

                # compute Hamming distance between last two given hints
                hamming_distance = self.reward_metrics.compute_hamming_distance(vectorized, self.vectorized_observation_shape()[0])

                # in case the hint was no 'play'-hint or 'save'-hint, the reward will still be 0
                if reward != 0:
                    reward *= np.sqrt(np.sqrt(hamming_distance))
                else:
                    reward= 0.01*np.sqrt(np.sqrt(hamming_distance))
                # update last action in reward storage
                self.reward_metrics.update_history(action, vectorized)  # used for next hamming distance
            elif action.type() == PLAY:
                # get pyhnabi.HanabiCard object for played card
                idx_card_played = action.card_index()
                card_played = self.state.player_hands()[cur_player][idx_card_played]
                if card_played.rank() in [2, 3, 4]:
                    if card_played.rank() == fireworks[card_played.color()] and fireworks[0] > 0 and fireworks[1] > 0:
                        reward = 2 ** card_played.rank()
        # ################################################ #
        # -------------- Custom Reward END --------------- #
        # ################################################ #


        last_score = self.state.score()

        # Apply the action to the state.
        self.state.apply_move(action)

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        observation = self._make_observation_all_players()
        done = self.state.is_terminal()

        # if reward was not modified, default to standard reward
        if reward is None:
            reward = self.state.score() - last_score

        info = {}  # todo add default reward here for comparison

        return (observation, reward, done, info)

    def _make_observation_all_players(self):
        """Make observation for all players.

        Returns:
          dict, containing observations for all players.
        """
        obs = {}
        player_observations = [self._extract_dict_from_backend(
            player_id, self.state.observation(player_id))
            for player_id in range(self.players)]  # pylint: disable=bad-continuation
        obs["player_observations"] = player_observations
        obs["current_player"] = self.state.cur_player()
        return obs

    def _extract_dict_from_backend(self, player_id, observation):
        """Extract a dict of features from an observation from the backend.

        Args:
          player_id: Int, player from whose perspective we generate the observation.
          observation: A `pyhanabi.HanabiObservation` object.

        Returns:
          obs_dict: dict, mapping from HanabiObservation to a dict.
        """
        obs_dict = {}
        obs_dict["current_player"] = self.state.cur_player()
        obs_dict["current_player_offset"] = observation.cur_player_offset()
        obs_dict["life_tokens"] = observation.life_tokens()
        obs_dict["information_tokens"] = observation.information_tokens()
        obs_dict["num_players"] = observation.num_players()
        obs_dict["deck_size"] = observation.deck_size()

        obs_dict["fireworks"] = {}
        fireworks = self.state.fireworks()
        for color, firework in zip(pyhanabi.COLOR_CHAR, fireworks):
            obs_dict["fireworks"][color] = firework

        obs_dict["legal_moves"] = []
        obs_dict["legal_moves_as_int"] = []
        for move in observation.legal_moves():
            obs_dict["legal_moves"].append(move.to_dict())
            obs_dict["legal_moves_as_int"].append(self.game.get_move_uid(move))

        obs_dict["observed_hands"] = []
        for player_hand in observation.observed_hands():
            cards = [card.to_dict() for card in player_hand]
            obs_dict["observed_hands"].append(cards)

        obs_dict["discard_pile"] = [
            card.to_dict() for card in observation.discard_pile()
        ]

        # Return hints received.
        obs_dict["card_knowledge"] = []
        for player_hints in observation.card_knowledge():
            player_hints_as_dicts = []
            for hint in player_hints:
                hint_d = {}
                if hint.color() is not None:
                    hint_d["color"] = pyhanabi.color_idx_to_char(hint.color())
                else:
                    hint_d["color"] = None
                hint_d["rank"] = hint.rank()
                player_hints_as_dicts.append(hint_d)
            obs_dict["card_knowledge"].append(player_hints_as_dicts)

        # ipdb.set_trace()
        obs_dict["vectorized"] = self.observation_encoder.encode(observation)
        obs_dict["pyhanabi"] = observation

        return obs_dict

    def _build_move(self, action):
        """Build a move from an action dict.

        Args:
          action: dict, mapping to a legal action taken by an agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }

        Returns:
          move: A `HanabiMove` object constructed from action.

        Raises:
          ValueError: Unknown action type.
        """
        assert isinstance(action, dict), "Expected dict, got: {}".format(action)
        assert "action_type" in action, ("Action should contain `action_type`. "
                                         "action: {}").format(action)
        action_type = action["action_type"]
        assert (action_type in MOVE_TYPES), (
            "action_type: {} should be one of: {}".format(action_type, MOVE_TYPES))

        if action_type == "PLAY":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_play_move(card_index=card_index)
        elif action_type == "DISCARD":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_discard_move(card_index=card_index)
        elif action_type == "REVEAL_RANK":
            target_offset = action["target_offset"]
            rank = action["rank"]
            move = pyhanabi.HanabiMove.get_reveal_rank_move(
                target_offset=target_offset, rank=rank)
        elif action_type == "REVEAL_COLOR":
            target_offset = action["target_offset"]
            assert isinstance(action["color"], str)
            color = color_char_to_idx(action["color"])
            move = pyhanabi.HanabiMove.get_reveal_color_move(
                target_offset=target_offset, color=color)
        else:
            raise ValueError("Unknown action_type: {}".format(action_type))

        legal_moves = self.state.legal_moves()
        assert (str(move) in map(
            str,
            legal_moves)), "Illegal action: {}. Move should be one of : {}".format(
            move, legal_moves)

        return move


def make(environment_name="Hanabi-Full", num_players=2, pyhanabi_path=None):
    """Make an environment.

    Args:
      environment_name: str, Name of the environment to instantiate.
      num_players: int, Number of players in this game.
      pyhanabi_path: str, absolute path to header files for c code linkage.

    Returns:
      env: An `Environment` object.

    Raises:
      ValueError: Unknown environment name.
    """

    if pyhanabi_path is not None:
        prefixes = (pyhanabi_path,)
        assert pyhanabi.try_cdef(prefixes=prefixes), "cdef failed to load"
        assert pyhanabi.try_load(prefixes=prefixes), "library failed to load"

    if (environment_name == "Hanabi-Full" or
            environment_name == "Hanabi-Full-CardKnowledge"):
        return HanabiEnv(
            config={
                "colors":
                    5,
                "ranks":
                    5,
                "players":
                    num_players,
                "max_information_tokens":
                    8,
                "max_life_tokens":
                    3,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
    elif environment_name == "Hanabi-Full-Minimal":
        return HanabiEnv(
            config={
                "colors": 5,
                "ranks": 5,
                "players": num_players,
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "observation_type": pyhanabi.AgentObservationType.MINIMAL.value
            })
    elif environment_name == "Hanabi-Small":
        return HanabiEnv(
            config={
                "colors":
                    2,
                "ranks":
                    5,
                "players":
                    num_players,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
    elif environment_name == "Hanabi-Very-Small":
        return HanabiEnv(
            config={
                "colors":
                    1,
                "ranks":
                    5,
                "players":
                    num_players,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
    else:
        raise ValueError("Unknown environment {}".format(environment_name))


# -------------------------------------------------------------------------------
# Hanabi Agent API
# -------------------------------------------------------------------------------


class Agent(object):
    """Agent interface.

    All concrete implementations of an Agent should derive from this interface
    and implement the method stubs.


    ```python

    class MyAgent(Agent):
      ...

    agents = [MyAgent(config) for _ in range(players)]
    while not done:
      ...
      for agent_id, agent in enumerate(agents):
        action = agent.act(observation)
        if obs.current_player == agent_id:
          assert action is not None
        else
          assert action is None
      ...
    ```
    """

    def __init__(self, config, *args, **kwargs):
        r"""Initialize the agent.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0)
              - max_life_tokens: int, Number of life tokens (>=0)
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
          *args: Optional arguments
          **kwargs: Optional keyword arguments.

        Raises:
          AgentError: Custom exceptions.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def reset(self, config):
        r"""Reset the agent with a new config.

        Signals agent to reset and restart using a config dict.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values.
              - colors: int, Number of colors \in [2,5].
              - ranks: int, Number of ranks \in [2,5].
              - players: int, Number of players \in [2,5].
              - hand_size: int, Hand size \in [4,5].
              - max_information_tokens: int, Number of information tokens (>=0)
              - max_life_tokens: int, Number of life tokens (>=0)
              - seed: int, Random seed.
              - random_start_player: bool, Random start player.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def act(self, observation):
        """Act based on an observation.

        Args:
          observation: dict, containing observation from the view of this agent.
            An example:
            {'current_player': 0,
             'current_player_offset': 1,
             'deck_size': 40,
             'discard_pile': [],
             'fireworks': {'B': 0,
                       'G': 0,
                       'R': 0,
                       'W': 0,
                       'Y': 0},
             'information_tokens': 8,
             'legal_moves': [],
             'life_tokens': 3,
             'observed_hands': [[{'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1},
                             {'color': None, 'rank': -1}],
                            [{'color': 'W', 'rank': 2},
                             {'color': 'Y', 'rank': 4},
                             {'color': 'Y', 'rank': 2},
                             {'color': 'G', 'rank': 0},
                             {'color': 'W', 'rank': 1}]],
             'num_players': 2}]}

        Returns:
          action: dict, mapping to a legal action taken by this agent. The following
            actions are supported:
              - { 'action_type': 'PLAY', 'card_index': int }
              - { 'action_type': 'DISCARD', 'card_index': int }
              - {
                  'action_type': 'REVEAL_COLOR',
                  'color': str,
                  'target_offset': int >=0
                }
              - {
                  'action_type': 'REVEAL_RANK',
                  'rank': str,
                  'target_offset': int >=0
                }
        """
        raise NotImplementedError("Not implemented in Abstract Base class")
