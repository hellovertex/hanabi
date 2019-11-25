import numpy as np
def set_game_vars(environment_name = 'Hanabi-Small3l', num_players = 2):
    
    global NUM_RANKS
    NUM_RANKS = 5
    
    global COLORS_ORDER, HAND_SIZE, NUM_PLAYERS, LIFE_TOKENS, INFO_TOKENS
    NUM_PLAYERS = num_players
    if 'Hanabi-Small' in environment_name:
        COLORS_ORDER = ['R', 'Y']
        HAND_SIZE = 2
        INFO_TOKENS = 3
        if '3l' in environment_name:
            LIFE_TOKENS = 3
        elif '4l' in environment_name:
            LIFE_TOKENS = 4
        elif '2l' in environment_name:
            LIFE_TOKENS = 2
        else:
            LIFE_TOKENS = 1
    elif 'Hanabi-Very-Small' in environment_name:
        COLORS_ORDER = ['R']
        HAND_SIZE = 2
        LIFE_TOKENS = 1
        INFO_TOKENS = 1
    elif 'Hanabi-Full' in environment_name:
        COLORS_ORDER = ["R", "Y", "G", "W", "B"]
        HAND_SIZE = 5
        LIFE_TOKENS = 3
        INFO_TOKENS = 8
    else:
        raise 'Unknown game type %s' % environment_name
    
    global NUM_COLORS
    NUM_COLORS = len(COLORS_ORDER)
    global BITS_PER_CARD
    BITS_PER_CARD = len(COLORS_ORDER) * NUM_RANKS
    
    global INITIAL_CARD_COUNT
    INITIAL_CARD_COUNT = np.zeros(BITS_PER_CARD, dtype = 'int')
    for color_ind in range(len(COLORS_ORDER)):
        INITIAL_CARD_COUNT[color_ind * NUM_RANKS + 0] = 3
        INITIAL_CARD_COUNT[color_ind * NUM_RANKS + 1] = 2
        INITIAL_CARD_COUNT[color_ind * NUM_RANKS + 2] = 2
        INITIAL_CARD_COUNT[color_ind * NUM_RANKS + 3] = 2
        INITIAL_CARD_COUNT[color_ind * NUM_RANKS + 4] = 1
        
    global TOTAL_CARDS
    TOTAL_CARDS = int(np.sum(INITIAL_CARD_COUNT))
    
    global HINT_MASK
    global NUM_ACTIONS
    NUM_ACTIONS = HAND_SIZE + HAND_SIZE + NUM_COLORS * (num_players - 1) + NUM_RANKS * (num_players - 1)
    HINT_MASK = np.zeros(NUM_ACTIONS, dtype = 'int')
    HINT_MASK[2 * HAND_SIZE : 2 * HAND_SIZE + NUM_COLORS * (num_players - 1)] = 1
    HINT_MASK[2 * HAND_SIZE + NUM_COLORS * (num_players - 1) :] = 2
    return (NUM_RANKS, NUM_COLORS, COLORS_ORDER, HAND_SIZE, 
            BITS_PER_CARD, INITIAL_CARD_COUNT, TOTAL_CARDS, HINT_MASK)

def set_indexes():
    
    global OBSERVED_HANDS_START, OBSERVED_HANDS_END
    
    OBSERVED_HANDS_START = 0
    OBSERVED_HANDS_END = (NUM_PLAYERS - 1) * HAND_SIZE * BITS_PER_CARD
    
    global FIREWORKS_START, FIREWORKS_END
    
    FIREWORKS_START = int(OBSERVED_HANDS_END + NUM_PLAYERS + TOTAL_CARDS - NUM_PLAYERS * HAND_SIZE)
    FIREWORKS_END = int(FIREWORKS_START + NUM_RANKS * NUM_COLORS)
    
    global DISCARD_START, DISCARD_END
    
    DISCARD_START = FIREWORKS_END + LIFE_TOKENS + INFO_TOKENS
    DISCARD_END = DISCARD_START + TOTAL_CARDS
    
    global LAST_ACTION_START, LAST_ACTION_END
    
    LAST_ACTION_START = DISCARD_END
    LAST_ACTION_END = LAST_ACTION_START + NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS + NUM_RANKS \
                      + HAND_SIZE + HAND_SIZE + BITS_PER_CARD + 2

game_type = 'Hanabi-Small'
NUM_PLAYERS = 2

(NUM_RANKS, NUM_COLORS, COLORS_ORDER, HAND_SIZE,
 BITS_PER_CARD, INITIAL_CARD_COUNT, TOTAL_CARDS, HINT_MASK) = set_game_vars(game_type, NUM_PLAYERS)
set_indexes()