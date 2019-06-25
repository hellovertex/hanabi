import pyhanabi_to_gui
import utils
# example observation for 2 players
observations = {'player_observations': [
    {'current_player': 0, 'current_player_offset': 0, 'life_tokens': 3, 'information_tokens': 8, 'num_players': 2,
     'deck_size': 40, 'fireworks': {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0},
     'legal_moves': [{'action_type': 'PLAY', 'card_index': 0}, {'action_type': 'PLAY', 'card_index': 1},
                     {'action_type': 'PLAY', 'card_index': 2}, {'action_type': 'PLAY', 'card_index': 3},
                     {'action_type': 'PLAY', 'card_index': 4},
                     {'action_type': 'REVEAL_COLOR', 'target_offset': 1, 'color': 'Y'},
                     {'action_type': 'REVEAL_COLOR', 'target_offset': 1, 'color': 'G'},
                     {'action_type': 'REVEAL_COLOR', 'target_offset': 1, 'color': 'W'},
                     {'action_type': 'REVEAL_RANK', 'target_offset': 1, 'rank': 0},
                     {'action_type': 'REVEAL_RANK', 'target_offset': 1, 'rank': 2}],
     'legal_moves_as_int': [5, 6, 7, 8, 9, 11, 12, 13, 15, 17], 'observed_hands': [
        [{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1},
         {'color': None, 'rank': -1}, {'color': None, 'rank': -1}],
        [{'color': 'Y', 'rank': 0}, {'color': 'W', 'rank': 2}, {'color': 'G', 'rank': 0}, {'color': 'G', 'rank': 0},
         {'color': 'Y', 'rank': 2}]], 'discard_pile': [], 'card_knowledge': [
        [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None},
         {'color': None, 'rank': None}, {'color': None, 'rank': None}],
        [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None},
         {'color': None, 'rank': None}, {'color': None, 'rank': None}]],
     'vectorized': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     'last_moves': []},
    {'current_player': 0, 'current_player_offset': 1, 'life_tokens': 3, 'information_tokens': 8, 'num_players': 2,
     'deck_size': 40, 'fireworks': {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}, 'legal_moves': [],
     'legal_moves_as_int': [], 'observed_hands': [
        [{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1},
         {'color': None, 'rank': -1}, {'color': None, 'rank': -1}],
        [{'color': 'G', 'rank': 2}, {'color': 'B', 'rank': 1}, {'color': 'G', 'rank': 0}, {'color': 'Y', 'rank': 2},
         {'color': 'R', 'rank': 4}]], 'discard_pile': [], 'card_knowledge': [
        [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None},
         {'color': None, 'rank': None}, {'color': None, 'rank': None}],
        [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None},
         {'color': None, 'rank': None}, {'color': None, 'rank': None}]],
     'vectorized': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     'last_moves': []}], 'current_player': 0}

"""
    # ################################################ #
    # ----------- EXAMPLE SERVER MESSAGES  ----------- #
    # ################################################ #
The order is the same when playing a game, 
i.e. first comes the init message, then the notifyList, etc.
"""

# example init message
msg_init = 'init {' \
           '"names":["SimpleAgent00","SimpleAgent01","test"],' \
           '"variant":"No Variant",' \
           '"seat":2,' \
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
           '"pausePlayer":"SimpleAgent00","pauseQueued":false}'


# example notifyList message
msg_notifyList = 'notifyList [' \
                 '{"type":"draw","who":0,"rank":3,"suit":3,"order":0},' \
                 '{"type":"draw","who":0,"rank":5,"suit":2,"order":1},' \
                 '{"type":"draw","who":0,"rank":4,"suit":4,"order":2},' \
                 '{"type":"draw","who":0,"rank":1,"suit":4,"order":3},' \
                 '{"type":"draw","who":0,"rank":1,"suit":2,"order":4},' \
                 '{"type":"draw","who":1,"rank":5,"suit":3,"order":5},' \
                 '{"type":"draw","who":1,"rank":4,"suit":2,"order":6},' \
                 '{"type":"draw","who":1,"rank":2,"suit":4,"order":7},' \
                 '{"type":"draw","who":1,"rank":1,"suit":2,"order":8},' \
                 '{"type":"draw","who":1,"rank":3,"suit":0,"order":9},' \
                 '{"type":"draw","who":2,"rank":-1,"suit":-1,"order":10},' \
                 '{"type":"draw","who":2,"rank":-1,"suit":-1,"order":11},' \
                 '{"type":"draw","who":2,"rank":-1,"suit":-1,"order":12},' \
                 '{"type":"draw","who":2,"rank":-1,"suit":-1,"order":13},' \
                 '{"type":"draw","who":2,"rank":-1,"suit":-1,"order":14},' \
                 '{"type":"status","clues":8,"score":0,"maxScore":25,"doubleDiscard":false},' \
                 '{"type":"text","text":"test goes first"},{"type":"turn","num":0,"who":2}]'

"""
    # ################################################ #
    # --------------------- TESTS  ------------------- #
    # ################################################ #
"""
# ( )
def test_msg_deal_card():
    observed_hands_0 = [
        [
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1}
        ],
        [
            {'color': 'Y', 'rank': 0},
            {'color': 'W', 'rank': 2},
            {'color': 'G', 'rank': 0},
            {'color': 'G', 'rank': 0},
            {'color': 'Y', 'rank': 2}
        ]
    ]
    observed_hands_1 = [
        [
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1},
            {'color': None, 'rank': -1}
        ],
        [
            {'color': 'G', 'rank': 2},
            {'color': 'B', 'rank': 1},
            {'color': 'G', 'rank': 0},
            {'color': 'Y', 'rank': 2},
            {'color': 'R', 'rank': 4}
        ]
    ]



# (x)
def test_format_names():
    names = [0,1,2]
    test_names_formatted = '"names":["0","1","2"]'
    result = pyhanabi_to_gui.format_names(names)
    #print(test_names_formatted, result)
    #print(test_names_formatted == result)
    assert test_names_formatted == result


# (x)
def test_create_init_message():
    agent_id = 1
    player_observation = observations['player_observations'][agent_id]

    test_msg = 'init {' \
               '"names":["0","1"],' \
               '"variant":"No Variant",' \
               '"seat":1,' \
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
               '"pausePlayer":"0","pauseQueued":false}'


    result = pyhanabi_to_gui.create_init_message(player_observation, agent_id)
    print(test_msg == result)

    assert test_msg == result


# (x)
def test_format_draw():
    cards = [{'color': None, 'rank': None}, {'color': 'Y', 'rank': 2}]
    test_msgs = [
        '{"type":"draw","who":0,"rank":-1,"suit":-1,"order":22}',
        '{"type":"draw","who":0,"rank":3,"suit":2,"order":22}'
    ]
    who = 0
    order = 22
    for card, test_msg in list(zip(cards, test_msgs)):
        result = pyhanabi_to_gui.format_draw(card, who, order)
        print(result, test_msg)
        assert result == test_msg


# (x)
def test_format_status():
    clues = 8
    score = 0
    max_score = 25

    test_msg = '{"type":"status","clues":8,"score":0,"maxScore":25,"doubleDiscard":false}'
    result = pyhanabi_to_gui.format_status(clues, score, max_score)

    #print(test_msg, result)
    assert test_msg == result


# (x)
def test_format_text_who_goes_first():
    name = str(0)
    pos_abs = 1
    test_msg = '{"type":"text","text":"0 goes first"},{"type":"turn","num":0,"who":1}]'
    result = pyhanabi_to_gui.format_text_who_goes_first(name, pos_abs)
    assert result == test_msg


# (x)
def test_from_relative_to_abs():
    agent_id = 1
    player_observation = observations['player_observations'][agent_id]
    target = 'observed_hands'
    test_array = [  # note how we switched own cards to position 'agent_id'
        [{'color': 'G', 'rank': 2},
         {'color': 'B', 'rank': 1},
         {'color': 'G', 'rank': 0},
         {'color': 'Y', 'rank': 2},
         {'color': 'R', 'rank': 4}],
        [{'color': None, 'rank': -1},
         {'color': None, 'rank': -1},
         {'color': None, 'rank': -1},
         {'color': None, 'rank': -1},
         {'color': None, 'rank': -1}]
    ]
    result = pyhanabi_to_gui.from_relative_to_abs(player_observation, target=target)

    # print(result, test_array)
    assert result == test_array


# (x)
def test_create_notifyList_message():
    # created from player observation on top of file by manually replacing rank and suits
    test_msg = 'notifyList [' \
                     '{"type":"draw","who":0,"rank":-1,"suit":-1,"order":0},' \
                     '{"type":"draw","who":0,"rank":-1,"suit":-1,"order":1},' \
                     '{"type":"draw","who":0,"rank":-1,"suit":-1,"order":2},' \
                     '{"type":"draw","who":0,"rank":-1,"suit":-1,"order":3},' \
                     '{"type":"draw","who":0,"rank":-1,"suit":-1,"order":4},' \
                     '{"type":"draw","who":1,"rank":1,"suit":2,"order":5},' \
                     '{"type":"draw","who":1,"rank":3,"suit":4,"order":6},' \
                     '{"type":"draw","who":1,"rank":1,"suit":1,"order":7},' \
                     '{"type":"draw","who":1,"rank":1,"suit":1,"order":8},' \
                     '{"type":"draw","who":1,"rank":3,"suit":2,"order":9},' \
                     '{"type":"status","clues":8,"score":0,"maxScore":25,"doubleDiscard":false},' \
                     '{"type":"text","text":"0 goes first"},{"type":"turn","num":0,"who":0}]'
    agent_id = 0
    player_observation = observations['player_observations'][agent_id]
    result = pyhanabi_to_gui.create_notifyList_message(player_observation)

    print(test_msg, result)
    print(test_msg == result)
    assert test_msg == result


# (x)
def test_format_play():
    index = 0
    suit = 1
    rank = 1
    order = 11
    test_msg = '{"type":"play","which":{"index":0,"suit":1,"rank":1,"order":11}}'
    result = pyhanabi_to_gui.format_play(index, suit, rank, order)
    print(test_msg == result)
    assert test_msg == result


class GameStateWrapperTestMock:
    def __init__(self):
        # card_numbers are stored in reversed order
        self.card_numbers = [[9,8,7,6,5], [4,3,2,1,0]]

# (x)
def test_pyhanabi_card_index_to_gui_card_order():


    game_state_wrapper = GameStateWrapperTestMock()

    hand_size = 5
    card_index = 1
    players = [0, 1]
    test_nums = [1, 6]  # expected returns when players 0 and 1 play cards with pyhanabi card_index 2

    for pid, test_num in list(zip(players, test_nums)):
        result = pyhanabi_to_gui.pyhanabi_card_index_to_gui_card_oder(pid, card_index, game_state_wrapper)
        print(result == test_num)
        assert result == test_num


class LastMoveTestMock:
    def __init__(self, return_values=0, type=utils.HanabiMoveType.PLAY):
        self.ret_val = return_values
        self.type = type
    def player(self):
        return self.ret_val

    def move(self):
        ret_val = self.ret_val
        type = self.type
        class MoveTestMock:
            def __init__(self, return_values, type):
                self.return_value = return_values
                self.action_type = type
            def color(self):
                return self.return_value

            def rank(self):
                return self.return_value

            def card_index(self):
                return self.return_value

            def type(self):

                return self.action_type
        return MoveTestMock(ret_val, type)


# (x)
def test_get_json_params_for_play_or_discard():
    last_move = LastMoveTestMock(0)
    game_state_wrapper = GameStateWrapperTestMock()
    test_index, test_suite, test_rank, test_order = 0, 3, 1, 5  # values derived from last_move andd game_state_wrapper
    index, suite, rank, order = pyhanabi_to_gui.get_json_params_for_play_or_discard(game_state_wrapper, last_move)

    print( (test_index, test_suite, test_rank, test_order) == (index, suite, rank, order))
    assert (test_index, test_suite, test_rank, test_order) == (index, suite, rank, order)


# (x)
def test_create_notify_message_play():


    last_move = LastMoveTestMock(0)
    game_state_wrapper = GameStateWrapperTestMock()
    test_msg = '{"type":"play","which":{"index":0,"suit":3,"rank":1,"order":5}}'
    result = pyhanabi_to_gui.create_notify_message_play(game_state_wrapper, last_move)

    print(test_msg == result)
    assert test_msg == result

# (x)
def test_create_notify_message_discard():
    last_move = LastMoveTestMock(0, type=utils.HanabiMoveType.DISCARD)
    game_state_wrapper = GameStateWrapperTestMock()
    test_msg = '{"type":"discard","failed":false,"which":{"index":0,"suit":3,"rank":1,"order":5}}'
    result = pyhanabi_to_gui.create_notify_message_discard(game_state_wrapper, last_move)
    print(test_msg == result)
    assert test_msg == result

# (x)
def test_format_reveal_move():
    cluetype = 1
    cluevalue = 3
    giver = 0
    target = 1
    touched = [0]
    test_msg = '{"type": "clue", "clue": {"type": 1, "value": 3}, "giver": 0, "list": [0], "target": 1, "turn": 0}'
    result = pyhanabi_to_gui.format_reveal_move(cluetype, cluevalue, giver, touched, target)
    print(test_msg == result)
    assert test_msg == result

# test_format_names()
# test_create_init_message()
# test_format_draw()
# test_format_status()
# test_from_relative_to_abs()
# test_format_text_who_goes_first()
# test_create_notifyList_message()
# test_format_play()
# test_pyhanabi_card_index_to_gui_card_order()
# test_get_json_params_for_play_or_discard()
# test_create_notify_message_play()
# test_create_notify_message_discard()
# test_format_reveal_move()