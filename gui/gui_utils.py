from typing import Dict
import gui_vectorizer
import gui_agent
import json_to_pyhanabi
from typing import Optional, Set, List
import enum


# Just for convenience, copied from pyhanabi
class HanabiMoveType(enum.IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5


class GameVariants(enum.IntEnum):
    # number of colors in the deck
    THREE_SUITS = 3
    FOUR_SUITS = 4
    FIVE_SUITS = 5
    SIX_SUITS = 6


def variant_from_num_colors(num_colors: int) -> str:

    if num_colors == 3:
        return  "Three Suits"
    elif num_colors == 4:
        return "Four Suits"
    elif num_colors == 5:
        return "No Variant"
    elif num_colors == 6:
        return "Six Suits"


def num_colors_from_variant(variant: str) -> int:
    """ Takes game variant string as required by GUI server and returns corresponding number of colors"""
    if variant == "No Variant":
        return 5
    elif variant == "Three Suits":
        return 3
    elif variant == "Four Suits":
        return 4
    elif variant == "Five Suits":
        return 5
    elif variant == "Six Suits":
        return 6


def get_observation_size(pyhanabi_config):
    """ Returns the len of the vectorized observation """
    num_players = pyhanabi_config['players']  # number of players ingame
    num_colors = pyhanabi_config['colors']
    num_ranks = pyhanabi_config['ranks']
    hand_size = pyhanabi_config['hand_size']
    max_information_tokens = pyhanabi_config['max_life_tokens']
    max_life_tokens = pyhanabi_config['max_information_tokens']
    env = json_to_pyhanabi.create_env_mock(
        num_players=num_players,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=hand_size,
        max_information_tokens=max_information_tokens,
        max_life_tokens=max_life_tokens,
    )

    vec = gui_vectorizer.ObservationVectorizer(env)
    return vec.total_state_length


def get_num_actions(pyhanabi_config):
    """ total number of moves possible each turn (legal or not, depending on num_players and num_cards),
    i.e. MaxDiscardMoves + MaxPlayMoves + MaxRevealColorMoves + MaxRevealRankMoves """
    num_players = pyhanabi_config["players"]
    hand_size = 4 if num_players > 3 else 5
    num_colors = pyhanabi_config['colors']
    num_ranks = pyhanabi_config['ranks']


    return 2 * hand_size + (num_players - 1) * num_colors + (num_players - 1) * num_ranks


def pyhanabi_color_to_gui_suit(color: str) -> Optional[int]:
    """
    Returns format desired by server
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
    """
    if color is None: return -1
    if color == 'B': return 0
    if color == 'G': return 1
    if color == 'Y': return 2
    if color == 'R': return 3
    if color == 'W': return 4
    return -1


def gui_suit_to_pyhanabi_color(suit: int) -> Optional[str]:

    """
    Returns format desired by pyhanabi
    // 0 is blue
    // 1 is green
    // 2 is yellow
    // 3 is red
    // 4 is purple
    returns None if suit is None or -1
    """
    if suit == -1: return None
    if suit == 0: return 'B'
    if suit == 1: return 'G'
    if suit == 2: return 'Y'
    if suit == 3: return 'R'
    if suit == 4: return 'W'
    return None


def sort_colors_to_pyhanabi_RYGWB(colors: Set) -> List:
    """ Sorts list, s.t. colors are in order RYGWB """
    result = list()
    for i in range(len(colors)):
        if 'R' in colors:
            colors.remove('R')
            result.append('R')
        if 'Y' in colors:
            colors.remove('Y')
            result.append('Y')
        if 'G' in colors:
            colors.remove('G')
            result.append('G')
        if 'W' in colors:
            colors.remove('W')
            result.append('W')
        if 'B' in colors:
            colors.remove('B')
            result.append('B')

    return result


def parse_rank_server(rank):
    """ Returns rank as expected by the gui """
    if int(rank) > -1:
            rank += 1

    return str(rank)