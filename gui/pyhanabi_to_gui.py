from typing import List, Optional, Dict, Tuple
import utils
import enum
""" This module parses actions provided by the pyhanabi interface from 
https://github.com/deepmind/hanabi-learning-environment such that they can be sent to the gui client, mirroring 
the behaviour of our gui server """




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
    Part of json message sent by the gui when setting up the game

    :param names: Array of integers (agent_ids)
    :return: Formatted array string containing the list of all players
    """
    assert len(names) >= 2

    ret = '"names":['
    for i in names:
        ret += f'"{i}",'
    # replace last ',' with ']'
    ret = ret[:-1] + ']'

    return ret


def format_draw(card: Dict, who: int, order: int) -> Optional[str]:
    """ Json message server sends when card is dealt """
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
    """ Part of json message sent by the gui when setting up the game """
    ret_msg = '{{"type":"status","clues":{0},"score":{1},"maxScore":{2},"doubleDiscard":false}}'.format(
        clues,
        score,
        max_score
    )
    return ret_msg


def format_text_who_goes_first(name, pos_abs):
    return '{{"type":"text","text":"{0} goes first"}},{{"type":"turn","num":0,"who":{1}}}]'.format(name, pos_abs)


def format_play(index, suit, rank, order):
    """
    Message sent on PLAY action, e.g.
    {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
     """
    return '{{"type":"play","which":{{"index":{},"suit":{},"rank":{},"order":{}}}}}'.format(
        index,
        suit,
        rank,
        order)


def format_discard(index, suit, rank, order):
    """
        Message sent on DISCARD action, e.g.
        {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
         """
    return '{{"type":"discard","failed":false,"which":{{"index":{},"suit":{},"rank":{},"order":{}}}}}'.format(
        index,
        suit,
        rank,
        order)


def format_reveal_move(cluetype, cluevalue, giver, touched, target):
    """
        Message sent on REVEAL_XYZ action, e.g.
        {"type": "clue", "clue": {"type": 1, "value": 3}, "giver": 0, "list": [5, 8, 9], "target": 1, "turn": 0}
        Note: a card is "touched" if it is target of a given hint. cluetype is 0 for REVEAL_RANK and 1 for REVEAL_COLOR
     """

    return '{{"type": "clue", "clue": {{"type": {}, "value": {}}}, "giver": {}, "list": {}, "target": {}, "turn": 0}}'.format(
        cluetype,
        cluevalue,
        giver,
        touched,
        target
    )

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

def pyhanabi_color_idx_to_gui_suit(color: int) -> int:
    """ Returns color used in json encoded action read by gui server
    # in gui:
    // blue is 0
    // green is 1
    // yellow is 2
    // red is 3
    // white is 4
    # in pyhanabi
    // 0 is red
    // 1 is yellow
    // 2 is green
    // 3 is white
    // 4 is blue
    """
    # pyhanabi
    if color == -1: return color
    COLOR_CHAR = ["R", "Y", "G", "W", "B"]
    COLOR_IDX = [0, 1, 2 ,3 ,4]
    # gui
    SUIT_CHAR = ["B", "G", "Y", "R", "W"]

    assert color in COLOR_IDX

    return SUIT_CHAR.index(COLOR_CHAR[color])


def pyhanabi_rank_to_gui_rank(rank: int) -> int:
    """ Returns rank as expected by gui server, i.e. 1-indexed"""
    if rank in [0,1,2,3,4]:
        return rank + 1
    else:
        return rank


def pyhanabi_card_index_to_gui_card_order(player, card_index, game_state_wrapper):
    """
    Returns the absolute card number of the card with given card_index in player's hand
    :param player: agent_id
    :param card_index:
    :param game_state_wrapper: the game state wrapper corresponding to player
    :return: Number of the card ranging from 0 to (deck_size - 1)
    """
    hand_size = len(game_state_wrapper.card_numbers[player])
    # cards are in reverse order, so take the 'mirrored' index
    actual_idx = (hand_size - 1) - card_index
    print(f"CARD NUMBERS {game_state_wrapper.card_numbers}")
    print(f"PLAYER {player}")
    print(f"ACTUAL INDEX {actual_idx}")
    card_num = game_state_wrapper.card_numbers[player][actual_idx]

    return card_num


def get_json_params_for_play_or_discard(game_state_wrapper, last_move, agent_id):
    """
    Assigns values to json keys in one of
        {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
    or
        {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
    :param game_state_wrapper: Used to get value for "order" key
    :param last_move: pyhanabi HanabiHistoryItem object containing data for index, suit, and rank
    :return: 4-tuple (index, suit, rank, order) used in json encoding of action
    """
    # assign values to json keys (following the gui's naming conventions)
    # player index
    index = agent_id
    # color of played card
    suit = pyhanabi_color_idx_to_gui_suit(last_move.color())
    # rank+1 of played card
    rank = pyhanabi_rank_to_gui_rank(last_move.rank())
    # absolute number of played card
    card_index = last_move.move().card_index()
    order = pyhanabi_card_index_to_gui_card_order(agent_id, card_index, game_state_wrapper)

    return index, suit, rank, order


def get_json_cluevalue_from_last_move(cluetype, last_move):
    """
    Input: pyhanabi HanabiHistoryItem
    For REVEAL_RANK moves this method returns the rank as sent in json formatted server message
    For REVEAL_COLOR moves this method returns the color as sent in json formatted server message.
    """
    assert cluetype in [0,1]

    cluevalue = None

    if cluetype == 0:  # REVEAL_RANK
        cluevalue = pyhanabi_rank_to_gui_rank(last_move.move().rank())
    else:  # REVEAL_COLOR
        cluevalue = pyhanabi_color_idx_to_gui_suit(last_move.move().color())

    return cluevalue


def get_json_params_for_reveal_move(game_state_wrapper, last_move, agent_id):
    """
        Assigns values to json keys in one of
            {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
            for REVEAL_RANK moves ("clue" "type" is 0)
        or
            {"type":"clue","clue":{"type":1,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
            for REVEAL_COLOR moves ("clue" "type" is 1)
        :param game_state_wrapper: Used to get value for key "List" that contains indices of touched cards
        :param last_move: pyhanabi HanabiHistoryItem object containing data needed to compute keys
        "clue"["type"], "value" "giver" "target"
        :return: 5-tuple (cluetype, cluevalue, giver, touched, target) used in json encoding of action
    """

    cluetype = 0 if last_move.move().type()==utils.HanabiMoveType.REVEAL_RANK else 1
    cluevalue = get_json_cluevalue_from_last_move(cluetype, last_move)
    giver = agent_id
    target = (giver + last_move.move().target_offset()) % game_state_wrapper.num_players
    # a card is touched, if it is target of a given hint
    touched = [
        pyhanabi_card_index_to_gui_card_order(target, idx, game_state_wrapper) for idx in last_move.card_info_revealed()
    ]


    return cluetype, cluevalue, giver, touched, target
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

    # append json for game status
    clues = player_observation['information_tokens']
    score = 0
    for k,v in player_observation['fireworks'].items():
        score += v
    ret_msg += format_status(clues, score) + ','

    # append json for who starts
    pos_abs = get_player_index_abs(player_observation)
    ret_msg += format_text_who_goes_first(name=str(pos_abs), pos_abs=pos_abs)

    return ret_msg


def create_notify_message_play(game_state_wrapper, last_move, agent_id):
    """
    Returns json encoded action given HanabiHistoryItem last_move
    :param last_move: HanabiHistoryItem
    :return: json encoded action corresponding to last_move, e.g.
    {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
    """
    assert last_move.move().type() == utils.HanabiMoveType.PLAY

    index, suit, rank, order = get_json_params_for_play_or_discard(game_state_wrapper, last_move, agent_id)
    print(f"INDEX PLAY: {index}")
    return format_play(index, suit, rank, order)


def create_notify_message_discard(game_state_wrapper, last_move, agent_id):
    """
    Returns json encoded message the server would send to its client when announcing a DISCARD move.
    :param game_state_wrapper: Used for card lookup
    :param last_move: HanabiHistoryItem object that contains the necessary information from pyhanabi
    :return: json encoded DISCAD MESSAGE, e.g.
    {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
    note that when "failed" equals True, we have discarded through a PLAY move.
    """

    assert last_move.move().type() == utils.HanabiMoveType.DISCARD

    index, suit, rank, order = get_json_params_for_play_or_discard(game_state_wrapper, last_move, agent_id)
    print(f"INDEX DISCARD: {index}")
    return format_discard(index, suit, rank, order)


def create_notify_message_reveal_move(game_state_wrapper, last_move, agent_id):
    """
    Returns json encoded message the server would send to its client when announcing a REVEAL_XYZ move.
    :param game_state_wrapper: Used for card lookup
    :param last_move: HanabiHistoryItem object that contains the necessary information from pyhanabi
    :return: json encoded REVEAL MESSAGE, e.g.
    a = {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
    Note that a["clue"]["type"] is 0 for REVEAL_RANK and 1 for REVEAL_COLOR
    "giver" and "target" are absolute player indices
    """

    assert last_move.move().type() in [utils.HanabiMoveType.REVEAL_COLOR, utils.HanabiMoveType.REVEAL_RANK]

    cluetype, cluevalue, giver, touched, target = get_json_params_for_reveal_move(game_state_wrapper, last_move, agent_id)
    print(f"GIVER REVEAL: {giver}")
    return format_reveal_move(cluetype, cluevalue, giver, touched, target)


def create_notify_message_from_last_move(game_state_wrapper, last_moves: List, agent_id: int) -> str:
    """


    Returns json encoded action as sent by gui server using the corresponding HanabiHistoryItem.
    We update the pyhanabi env first and then call this method here to sync the gui clients wrapped envs,
    before we start the next iteration of the game loop. This is for convenience, as the
    HanabiHistoryItem contains all the information, we otherwise had to compute from within
    the GameStateWrapper
    :param last_moves: list of HanabiHistoryItem s
    :return: json encoded action for last move
    (used for updating internal state of guis game_state_wrapper)
    """
    ret_msg = 'notify '

    assert len(last_moves) > 0
    # IGNORE DEAL MOVES
    if last_moves[0].move().type() == utils.HanabiMoveType.DEAL:

        assert len(last_moves) > 1
        last_move = last_moves[1]  # we never see 2 consecutive DEAL moves in last_moves
    else:
        last_move = last_moves[0]

    # CASE 1: PLAY MOVE
    if last_move.move().type() == utils.HanabiMoveType.PLAY:
        ret_msg += create_notify_message_play(game_state_wrapper, last_move, agent_id)

    # CASE 2: DISCARD MOVE
    elif last_move.move().type() == utils.HanabiMoveType.DISCARD:
        ret_msg += create_notify_message_discard(game_state_wrapper, last_move, agent_id)

    # CASE 3: REVEAL_XYZ
    elif last_move.move().type() in [utils.HanabiMoveType.REVEAL_COLOR, utils.HanabiMoveType.REVEAL_RANK]:
        ret_msg += create_notify_message_reveal_move(game_state_wrapper, last_move, agent_id)

    else:
        return None

    return ret_msg


def get_json_params_for_deal_move(game_state_wrapper, last_move, agent_id):
    """ Returns keys "who", "rank", "suit", "order" for json encoded DEAL move as sent by gui server """
    who = agent_id
    rank = pyhanabi_rank_to_gui_rank(last_move.move().rank())
    suit = pyhanabi_color_idx_to_gui_suit(last_move.move().rank())
    order = game_state_wrapper.order  # contains number of card drawn next

    return who, rank, suit, order


def create_notify_message_deal(game_state_wrapper, last_moves, agent_id):
    """ Converts pyhanabis DEAL action from last_move to json encoded DEAL action as sent by gui server
    e.g. {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
    """
    assert last_moves[0].move().type() == utils.HanabiMoveType.DEAL

    ret_msg = 'notify '

    who, rank, suit, order = get_json_params_for_deal_move(game_state_wrapper, last_moves[0], agent_id)

    card = {'color': suit, 'rank': rank}
    ret_msg += format_draw(card, who, order)

    return ret_msg