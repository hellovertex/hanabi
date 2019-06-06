import os
import sys

import numpy as np
import run_experiment as xp
import vectorizer
import agent_player as ap

def get_mock_observation_3rd_state_2pl():
# Life Tokens: 3
# Info tokens: 7
# Fireworks: R0 Y0 G0 W1 B0
# Hands:
# Cur player
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# -----
# R4 || XX|RYGB12345
# G2 || XX|RYGB12345
# Y1 || XX|RYGB12345
# G3 || XX|RYGB12345
# W1 || XX|RYGWB12345
# Deck size: 39
# Discards: []
#
# Current Agent Player Action: {'color': 'Y', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}
# Agent: 0 action: {'color': 'Y', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}
# Last moves: [<(Deal W1)>, <(Play 0) by player 1 scored W1>, <(Reveal player +1 color W) by player 0 reveal 0>]

    vectorized = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 2
    life_tokens = 3
    information_tokens = 7
    deck_size = 39
    current_player = 0
    current_player_offset = 0

    legal_moves_as_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]

    legal_moves = [{'card_index': 0, 'action_type': 'DISCARD'}, {'card_index': 1, 'action_type': 'DISCARD'}, {'card_index': 2, 'action_type': 'DISCARD'}, {'card_index': 3, 'action_type': 'DISCARD'}, {'card_index': 4, 'action_type': 'DISCARD'}, {'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'card_index': 4, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'Y', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'G', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 1, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 0, 'B': 0, 'R': 0, 'W': 1, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'R', 'rank': 3}, {'color': 'G', 'rank': 1}, {'color': 'Y', 'rank': 0}, {'color': 'G', 'rank': 2}, {'color': 'W', 'rank': 0}]]

    discard_pile = []

    last_moves = []

    card_knowledge = [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': last_moves  # actually not contained in the returned dict of th
    }

    return observation


def get_mock_observation_2nd_state_2pl():


# Life tokens: 3
# Info tokens: 7
# Fireworks: R0 Y0 G0 W0 B0
# Hands:
# Cur player
# XX || WX|W12345
# XX || XX|RYGB12345
# XX || XX|RYGB12345
# XX || XX|RYGB12345
# XX || XX|RYGB12345
# -----
# R2 || XX|RYGWB12345
# B3 || XX|RYGWB12345
# B2 || XX|RYGWB12345
# R5 || XX|RYGWB12345
# W4 || XX|RYGWB12345
# Deck size: 40
# Discards:, 'discard_pile': [], 'card_knowledge': }
#
# End CURRENT PLAYER Observation
# LAST MOVES
# Current Agent Player Action: {'card_index': 0, 'action_type': 'PLAY'}
# Agent: 1 action: {'card_index': 0, 'action_type': 'PLAY'}

    # Full Game in mock_2pl_game.txt - last MOVE: Agent: 0 action: {'card_index': 0, 'action_type': 'PLAY'}
    vectorized = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 2
    life_tokens = 3
    information_tokens = 7
    deck_size = 40
    current_player = 0
    current_player_offset = 0

    legal_moves_as_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]

    legal_moves = [{'card_index': 0, 'action_type': 'DISCARD'}, {'card_index': 1, 'action_type': 'DISCARD'}, {'card_index': 2, 'action_type': 'DISCARD'}, {'card_index': 3, 'action_type': 'DISCARD'}, {'card_index': 4, 'action_type': 'DISCARD'}, {'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'card_index': 4, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 1, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 4, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 0, 'B': 0, 'R': 0, 'W': 0, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'R', 'rank': 1}, {'color': 'B', 'rank': 2}, {'color': 'B', 'rank': 1}, {'color': 'R', 'rank': 4}, {'color': 'W', 'rank': 3}]]

    discard_pile = []

    last_moves = []

    card_knowledge =  [[{'color': 'W', 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': last_moves  # actually not contained in the returned dict of th
    }

    return observation

def get_mock_observation_init_state_2pl():

# Life tokens: 3
# Info tokens: 8
# Fireworks: R0 Y0 G0 W0 B0
# Hands:
# Cur player
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# -----
# W1 || XX|RYGWB12345
# R4 || XX|RYGWB12345
# G2 || XX|RYGWB12345
# Y1 || XX|RYGWB12345
# G3 || XX|RYGWB12345
# Deck size: 40
# Discards:, 'discard_pile': [], 'card_knowledge': }
#
# Current Agent Player Action: {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}
# Agent: 0 action: {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}

#Last moves: [<(Reveal player +1 color W) by player 1 reveal 0>]

#Last player card knowledge: [[{'color': None, 'rank': None}, {'color': 'G', 'rank': None}, {'color': 'G', 'rank': None}, {'color': None, 'rank': None}, {'color': 'G', 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    # Full Game in mock_2pl_game.txt - last MOVE: Agent: 0 action: {'card_index': 0, 'action_type': 'PLAY'}
    vectorized = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 2
    life_tokens = 3
    information_tokens = 8
    deck_size = 40
    current_player = 0
    current_player_offset = 0

    legal_moves_as_int = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]

    legal_moves = [{'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'card_index': 4, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'Y', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'G', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 1, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 0, 'B': 0, 'R': 0, 'W': 0, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'W', 'rank': 0}, {'color': 'R', 'rank': 3}, {'color': 'G', 'rank': 1}, {'color': 'Y', 'rank': 0}, {'color': 'G', 'rank': 2}]]

    discard_pile = []
    last_moves = []
    card_knowledge =  [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': last_moves  # actually not contained in the returned dict of th
    }

    return observation

def get_mock_reveal_multiple():

# Life tokens: 3
# Info tokens: 4
# Fireworks: R0 Y1 G0 W1 B1
# Hands:
# Cur player
# XX || XX|R12345
# XX || GX|G12345
# XX || GX|G12345
# XX || XX|RW12345
# XX || GX|G12345
# -----
# R2 || XX|RYGWB12345
# B3 || XX|RYGWB12345
# B2 || XX|RYGWB12345
# R5 || XX|RYGWB12345
# W4 || XX|RYGWB12345
# Deck size: 37
# Discards: []
#
# End CURRENT PLAYER Observation
# Last move: <(Reveal player +1 color G) by player 1 reveal 1,2,4>
# Current Agent Player Action: {'card_index': 1, 'action_type': 'PLAY'}
# Agent: 1 action: {'card_index': 1, 'action_type': 'PLAY'}


    vectorized = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 2
    life_tokens = 3
    information_tokens = 4
    deck_size = 37
    current_player = 0
    current_player_offset = 0

    legal_moves_as_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]

    legal_moves = [{'card_index': 0, 'action_type': 'DISCARD'}, {'card_index': 1, 'action_type': 'DISCARD'}, {'card_index': 2, 'action_type': 'DISCARD'}, {'card_index': 3, 'action_type': 'DISCARD'}, {'card_index': 4, 'action_type': 'DISCARD'}, {'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'card_index': 4, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 1, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 4, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 1, 'B': 1, 'R': 0, 'W': 1, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'R', 'rank': 1}, {'color': 'B', 'rank': 2}, {'color': 'B', 'rank': 1}, {'color': 'R', 'rank': 4}, {'color': 'W', 'rank': 3}]]

    discard_pile = []
    last_moves = []
    card_knowledge =  [[{'color': None, 'rank': None}, {'color': 'G', 'rank': None}, {'color': 'G', 'rank': None}, {'color': None, 'rank': None}, {'color': 'G', 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': last_moves  # actually not contained in the returned dict of th
    }

    return observation

def get_mock_4pl_2nd_state():

# Life tokens: 3
# Info tokens: 7
# Fireworks: R0 Y0 G0 W0 B0
# Hands:
# Cur player
# XX || XX|RYWB12345
# XX || GX|G12345
# XX || XX|RYWB12345
# XX || GX|G12345
# -----
# R5 || XX|RYGWB12345
# R1 || XX|RYGWB12345
# W3 || XX|RYGWB12345
# B3 || XX|RYGWB12345
# -----
# B1 || XX|RYGWB12345
# Y1 || XX|RYGWB12345
# W4 || XX|RYGWB12345
# R3 || XX|RYGWB12345
# -----
# W1 || XX|RYGWB12345
# W5 || XX|RYGWB12345
# R4 || XX|RYGWB12345
# W2 || XX|RYGWB12345
# Deck size: 34
# last Move: [<(Reveal player +1 color G) by player 3 reveal 1,3>]
# Discards: []

    vectorized = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 4
    life_tokens = 3
    information_tokens = 7
    deck_size = 34
    current_player = 1
    current_player_offset = 0

    legal_moves_as_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 21, 23, 25, 27, 28, 30, 31, 33, 34, 36, 37]

    legal_moves = [{'card_index': 0, 'action_type': 'DISCARD'}, {'card_index': 1, 'action_type': 'DISCARD'}, {'card_index': 2, 'action_type': 'DISCARD'}, {'card_index': 3, 'action_type': 'DISCARD'}, {'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'R', 'target_offset': 2, 'action_type': 'REVEAL_COLOR'}, {'color': 'Y', 'target_offset': 2, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 2, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 2, 'action_type': 'REVEAL_COLOR'}, {'color': 'R', 'target_offset': 3, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 3, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 4, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 1, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 4, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 0, 'B': 0, 'R': 0, 'W': 0, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'R', 'rank': 4}, {'color': 'R', 'rank': 0}, {'color': 'W', 'rank': 2}, {'color': 'B', 'rank': 2}], [{'color': 'B', 'rank': 0}, {'color': 'Y', 'rank': 0}, {'color': 'W', 'rank': 3}, {'color': 'R', 'rank': 2}], [{'color': 'W', 'rank': 0}, {'color': 'W', 'rank': 4}, {'color': 'R', 'rank': 3}, {'color': 'W', 'rank': 1}]]

    discard_pile = []
    last_moves = []

    card_knowledge =  [[{'color': None, 'rank': None}, {'color': 'G', 'rank': None}, {'color': None, 'rank': None}, {'color': 'G', 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': last_moves  # actually not contained in the returned dict of th
    }

    return observation

def get_mock_4pl_3rd_state():
# Life tokens: 2
# Info tokens: 7
# Fireworks: R0 Y0 G0 W0 B0
# Hands:
# Cur player
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# XX || XX|RYGWB12345
# -----
# B1 || XX|RYGWB12345
# Y1 || XX|RYGWB12345
# W4 || XX|RYGWB12345
# R3 || XX|RYGWB12345
# -----
# W1 || XX|RYGWB12345
# W5 || XX|RYGWB12345
# R4 || XX|RYGWB12345
# W2 || XX|RYGWB12345
# -----
# B3 || XX|RYWB12345
# Y3 || XX|RYWB12345
# G1 || GX|G12345
# G4 || XX|RYGWB12345
# Deck size: 33
# Last ACTIONS: [<(Deal G4)>, <(Play 1) by player 3 G2>, <(Reveal player +1 color G) by player 2 reveal 1,3>]
# Discards: G2

    vectorized = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 4
    life_tokens = 2
    information_tokens = 7
    deck_size = 33
    current_player = 1
    current_player_offset = 0

    legal_moves_as_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 33, 35, 36]

    legal_moves = [{'card_index': 0, 'action_type': 'DISCARD'}, {'card_index': 1, 'action_type': 'DISCARD'}, {'card_index': 2, 'action_type': 'DISCARD'}, {'card_index': 3, 'action_type': 'DISCARD'}, {'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'Y', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'R', 'target_offset': 2, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 2, 'action_type': 'REVEAL_COLOR'}, {'color': 'Y', 'target_offset': 3, 'action_type': 'REVEAL_COLOR'}, {'color': 'G', 'target_offset': 3, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 3, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 1, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 2, 'rank': 4, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 3, 'rank': 3, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 0, 'B': 0, 'R': 0, 'W': 0, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'B', 'rank': 0}, {'color': 'Y', 'rank': 0}, {'color': 'W', 'rank': 3}, {'color': 'R', 'rank': 2}], [{'color': 'W', 'rank': 0}, {'color': 'W', 'rank': 4}, {'color': 'R', 'rank': 3}, {'color': 'W', 'rank': 1}], [{'color': 'B', 'rank': 2}, {'color': 'Y', 'rank': 2}, {'color': 'G', 'rank': 0}, {'color': 'G', 'rank': 3}]]

    discard_pile = [{'color': 'G', 'rank': 1}]

    last_moves = []

    card_knowledge =  [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': 'G', 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'legal_moves_as_int': legal_moves_as_int,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': last_moves  # actually not contained in the returned dict of th
    }

    return observation



if __name__=="__main__":
    ### Set up the environment
    game_type = "Hanabi-Full"
    num_players = 4
    env = xp.create_environment(game_type=game_type, num_players=num_players)


    # Setup vectorizer
    obs_vectorizer = vectorizer.ObservationVectorizer(env)

    # 2 Player 2nd state EDGE CASE TEST
    # ACTION: [<(Reveal player +1 color W) by player 1 reveal 0>]
    # mock_observation = get_mock_observation_2nd_state_2pl()
    # obs_vectorizer.last_player_card_knowledge =  [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]
    # obs_vectorizer.last_player_action = {"MOVE_TYPE":"REVEAL_COLOR", "PLAYER":1, "CARD_ID":None, "COLOR":"W", "RANK":None ,"TARGET_OFFSET": 0 , "SCORED":False, "INFO_ADD":False, "POSITIONS":[0]}

    # 2 Player 3rd state EDGE CASE TEST
    # Last moves: [<(Deal W1)>, <(Play 0) by player 1 scored W1>, <(Reveal player +1 color W) by player 0 reveal 0>]
    # mock_observation = get_mock_observation_3rd_state_2pl()
    # obs_vectorizer.last_player_card_knowledge = [{'color': 'W', 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]
    # obs_vectorizer.last_player_action = {"MOVE_TYPE":"PLAY", "PLAYER":1, "CARD_ID":0, "COLOR":"W", "RANK":0 ,"TARGET_OFFSET": None , "SCORED":True, "INFO_ADD":0}

    # 4 Player Edge Case - 2nd state
    # LAST ACTION: [<(Reveal player +1 color G) by player 3 reveal 1,3>]
    mock_observation = get_mock_4pl_2nd_state()
    obs_vectorizer.last_player_card_knowledge = [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]
    obs_vectorizer.last_player_action = {"action_type":"REVEAL_COLOR", "player":3, "hand_card_id":None, "color":"G", "RANK":None ,"target_offset": 1 , "scored":False, "info_add":False, "positions":[1,3]}

    # 4 Player Edge Case - 3rd state
    # LAST ACTION: [<(Deal G4)>, <(Play 1) by player 3 G2>, <(Reveal player +1 color G) by player 2 reveal 1,3>]
    # mock_observation = get_mock_4pl_3rd_state()
    # obs_vectorizer.last_player_card_knowledge = [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]
    # obs_vectorizer.last_player_action = {"action_type":"PLAY", "player":3, "hand_card_id": 1, "color":"G", "rank":1 ,"target_offset": 0, "scored": False, "info_add": None, "positions": None}
    #
    vectorized_obs_vectorizer = obs_vectorizer.vectorize_observation(mock_observation)
    vectorized_obs_mock = mock_observation["vectorized"]
    wrong_indices = np.where(np.equal(vectorized_obs_vectorizer, vectorized_obs_mock)*1 != 1)
    print("Wrong Indices: {}".format(wrong_indices))
    print("Length of wrong indices: {}\n".format(wrong_indices[0].shape))

    # DONE
    # TEST LEGAL_ACTIONS_TO_INT FUNCTION AND LOADING PRETRAINED TF MODEL

    # Setup Obs Stacker that keeps track of Observation for all agents ! Already includes logic for distinguishing the view between different agents
    # history_size = 1
    # obs_stacker = xp.create_obs_stacker(env,history_size=history_size)
    # observation_size = obs_stacker.observation_size()

    ### Set up the RL-Player, reload weights from trained model
    # agent = "DQN"
    ### Specify model weights to be loaded
    # path = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/experiments/full_4pl_2000it/playable_models"
    # iteration_no = 1950
    # player = ap.RLPlayer(agent,env,observation_size,history_size)
    # player.load_model_weights(path,iteration_no)
    # legal_moves_vectorizer = vectorizer.LegalMovesVectorizer(env)
    # legal_moves_as_int = legal_moves_vectorizer.legal_moves_to_int(mock_observation["legal_moves"])
    # print(np.equal(legal_moves_as_int,mock_observation["legal_moves_as_int"]))
