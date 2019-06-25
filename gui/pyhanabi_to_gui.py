from typing import List, Optional, Dict, Tuple
import utils
import enum

class HanabiMoveType(enum.IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5


""" 
EXAMPLE FORMATS FOR JSON ENCODED ACTIONS: 

############   DRAW   ##############
{"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}

############   CLUE   ##############
{"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}

############   PLAY   ##############
{"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}

############   DISCARD   ##############
{"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
"""

"""
    # ################################################ #
    # ----------- JSON FORMATTING FUNCTIONs ---------- #
    # ################################################ #
"""

def format_names(names: List) -> str:
    """
    :param names: Array of integers (agent_ids)
    :return: Formatted string, that the gui server can read
    """
    assert len(names) >= 2

    ret = '"names":['
    for i in names:
        ret += f'"{i}",'
    # replace last ',' with ']'
    ret = ret[:-1] + ']'

    return ret


def format_draw(card: Dict, who: int, order: int) -> Optional[str]:
    def convert_rank(rank):
        if rank is None:
            return -1
        elif rank > -1:
            return rank + 1
        else:
            return rank
    suit = utils.convert_color(card['color'])
    rank = convert_rank(card['rank'])
    who = str(who)
    order = str(order)

    return '{{"type":"draw","who":{0},"rank":{1},"suit":{2},"order":{3}}}'.format(who, rank, suit, order)


def format_status(clues, score, max_score=25):
    ret_msg = '{{"type":"status","clues":{0},"score":{1},"maxScore":{2},"doubleDiscard":false}}'.format(
        clues,
        score,
        max_score
    )
    return ret_msg


def format_text_who_goes_first(name, pos_abs):
    return '{{"type":"text","text":"{0} goes first"}},{{"type":"turn","num":0,"who":{1}}}]'.format(name, pos_abs)

"""
    # ################################################ #
    # ------------ INDEX CONVERSION UTILS  ----------- #
    # ################################################ #
"""


def get_player_index_abs(player_observation):
    """
    Returns the absolute player position
    :param current_player: Not necessarily the calling agent
    :param current_player_offset: distance of calling agent to current_player
    :return: calling agents index (distance from player who starts)
    """
    num_players = player_observation['num_players']
    current_player = player_observation['current_player']
    current_player_offset = player_observation['current_player_offset']

    return (current_player + current_player_offset) % num_players


def get_player_list_abs(player_observation: List) -> Optional[List]:
    """ Returns player list (indices) relative to current player """
    current_player = player_observation['current_player']
    player_ids = [current_player]
    num_players = player_observation['num_players']

    for offset in range(1, len(player_observation)):
        # current player is the same for all obs at a fixed turn, just the offset changes
        player_ids.append((current_player + offset) % num_players)

    return player_ids


def from_relative_to_abs(player_observation, target='observed_hands'):
    """ Given player observation, shift the target array, s.t. the calling agent no longer sits at index 0
    Instead the birds eye perspective is returned
    """
    def shift(arr, n):
        return arr[n:] + arr[:n]

    p_idx = get_player_index_abs(player_observation)

    assert target in player_observation
    assert isinstance(player_observation[target], list)

    return shift(player_observation[target], p_idx)

# todo add yield to enforce order of calling these methods


"""
    # ################################################ #
    # ------------- CARD CONVERSION UTILS  ----------- #
    # ################################################ #
"""

def get_card_order_from_color_and_rank(game_state_wrapper, card):
    """
    Server handles internal card references by absolute card numbers, which are not provided by rl_env.
    We can only restore the card order by looking directly into the game_state_wrapper
    :param game_state_wrapper: Handles the gui environment. Each agent has its own game_state_wrapper
    :param card: pyhanabi HanabiCard object)
    returns: card_order, i.e. the number of the given card as stored inside the game_state_wrapper
     """

    # TODO
    # convert card to dictionary storing values as maintained by gui
    # then get index of this card by performing lookup in self.hand_list,
    # usinng this index, get abs card number by lookup in self.card_nums
    # return index

    pass
"""
    # ################################################ #
    # ---------- ACTION TO MESSAGE FUNCTIONs --------- #
    # ################################################ #
"""


def create_init_message(player_observation, agent_id):
    """
    Returns json that will be injected into game_state_wrapper.GameStateWrapper.init_players()
    This changes the state of the game state wrapper to after the game has started, from the
    perspective of the calling agent
    :param observation: pyhanabi observation of calling agent [NOT THE FULL OBS obj]
    :return: init_message: json encoded player list that can be read by server GUI
    """
    # names are gonna be the string values of the agent_id (which is just their index)

    num_players = player_observation['num_players']
    names = [str(i) for i in range(num_players)]
    names_formatted = format_names(names)

    return 'init {{' \
               '{0},' \
               '"variant":"No Variant",' \
               '"seat":{1},' \
               '"spectating":false,' \
               '"replay":false,' \
               '"sharedReplay":false,' \
               '"id":3,"timed":false,' \
               '"baseTime":0,' \
               '"timePerTurn":0,' \
               '"speedrun":false,' \
               '"deckPlays":false,' \
               '"emptyClues":false,' \
               '"characterAssignments":[],' \
               '"characterMetadata":[],' \
               '"correspondence":false,' \
               '"hypothetical":false,' \
               '"hypoActions":[],' \
               '"paused":false,' \
               '"pausePlayer":"0","pauseQueued":false}}'.format(names_formatted, agent_id)


def create_notifyList_message(player_observation):

    """
    Returns json that will be injected into game_state_wrapper.GameStateWrapper.deal_cards()
    This changes the state of the game state wrapper to after cards have been dealt, from the
    perspective of the calling agent
    :param observation: pyhanabi observation of calling agent [NOT THE FULL OBS obj]
    :return: init_message: json encoded player list that can be read by server GUI
    """
    ret_msg = 'notifyList ['

    # Server stores order to represent the absolute card number ranging from 0 to deck size-1
    order = 0
    # birds eye
    observed_hands_abs = from_relative_to_abs(player_observation, target='observed_hands')

    num_players = player_observation['observed_hands'][0]
    hand_size = len(num_players)

    # append jsons for card deals
    for pid in range(player_observation['num_players']):
        for cid in range(hand_size):
            ret_msg += format_draw(observed_hands_abs[pid][cid], who=pid, order=order) + ','
            order += 1
    #
    # append json for game status
    clues = player_observation['information_tokens']
    score = 0
    for k,v in player_observation['fireworks'].items():
        score += v
    ret_msg += format_status(clues, score) + ','

    # append json for who starts
    pos_abs = get_player_index_abs(player_observation)
    name = str(pos_abs)
    ret_msg += format_text_who_goes_first(name, pos_abs)

    return ret_msg


def _create_notify_message_play(game_state_wrapper, last_move):
    """
    Returns json encoded action given HanabiHistoryItem last_move
    :param last_move: HanabiHistoryItem
    :return: json encoded action corresponding to last_move, e.g.
    {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
    """
    assert last_move.move().type() == HanabiMoveType.PLAY

    # player index
    index = None
    # color of played card
    suit = None
    # rank+1 of played card
    rank = None
    # absolute number of played card
    order = None


def create_notify_message_from_last_move(game_state_wrapper, last_moves: List) -> str:
    """
    Returns json encoded action for server using the corresponding HanabiHistoryItem.
    We update the pyhanabi env first and then call this method here to sync the gui,
    before we start the next iteration of the game loop. This is for convenience, as the
    HanabiHistoryItem contains all the information, we otherwise had to compute from within
    the GameStateWrapper
    :param last_moves: list of HanabiHistoryItem s
    :return: json encoded action for last move
    (used for updating internal state of guis game_state_wrapper)
    """
    ret_msg = ''



    assert len(last_moves) > 0
    # IGNORE DEAL MOVES
    if last_moves[0].move().type() == HanabiMoveType.DEAL:
        assert len(last_moves) > 1
        last_move = last_moves[1]  # we never see 2 consecutive DEAL moves in last_moves
    else:
        last_move = last_moves[0]
    # CASE 1: PLAY MOVE
    if last_move.move().type() == HanabiMoveType.PLAY:
        ret_msg = _create_notify_message_play(game_state_wrapper, last_move)

    # CASE 2: DISCARD MOVE
    # CASE 3: REVEAL_RANK
    # CASE 4: REVEAL_COLOR

