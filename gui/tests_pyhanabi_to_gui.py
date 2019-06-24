import pyhanabi_to_gui
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
def test_get_player_list():

    test_player_ids = [0, 1]
    result = pyhanabi_to_gui.get_player_list(observations['player_observations'])
    print(test_player_ids == result) # suceeded
    assert test_player_ids == result


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


# test_get_player_list()
# test_format_names()
# test_create_init_message()
# test_format_draw()
# test_format_status()
# test_from_relative_to_abs()
# test_format_text_who_goes_first()
# test_create_notifyList_message()