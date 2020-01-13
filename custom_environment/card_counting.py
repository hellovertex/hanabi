import rl_env
import numpy as np
from PPOAgent.gamesettings import *

#PLAY = 1
#DISCARD = 2
#REVEAL_COLOR = 3
#REVEAL_RANK = 4
def get_mask_from_hint(obs_vec):
    last_action_type = get_last_action_type(obs_vec)
    last_action_vec = obs_vec[LAST_ACTION_START : LAST_ACTION_END]
    if last_action_type == 3:
        color_oh = last_action_vec[NUM_PLAYERS + 4 + NUM_PLAYERS : 
                                   NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS]
        color_ind = np.argmax(color_oh)
        mask = np.ones(BITS_PER_CARD, dtype = 'int')
        mask[0 : color_ind * NUM_RANKS] = mask[(color_ind + 1) * NUM_RANKS :] = 0
    elif last_action_type == 4:
        rank_oh = last_action_vec[NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS : 
                                  NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS + NUM_RANKS]
        rank = np.argmax(rank_oh)
        mask = np.zeros(BITS_PER_CARD, dtype = 'int')
        mask[rank::NUM_RANKS] = 1
    else:
        raise 'BadActionType %d' % (last_action_type)
    return mask

def get_last_action_type(obs_vec):
    last_action_vec = obs_vec[LAST_ACTION_START : LAST_ACTION_END]
    last_action_type_oh = last_action_vec[NUM_PLAYERS : NUM_PLAYERS + 4]
    return np.argmax(last_action_type_oh) + 1

def get_hint_offset(obs_vec):
    last_action_vec = obs_vec[LAST_ACTION_START : LAST_ACTION_END]
    offset_oh = last_action_vec[NUM_PLAYERS + 4 : NUM_PLAYERS + 4 + NUM_PLAYERS]
    offset = np.argmax(offset_oh)
    return offset
    
def get_cards_hinted_vec(obs_vec):
    last_action_vec = obs_vec[LAST_ACTION_START : LAST_ACTION_END]
    return last_action_vec[NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS + NUM_RANKS :
                           NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS + NUM_RANKS + HAND_SIZE]

def get_card_played(obs_vec):
    last_action_vec = obs_vec[LAST_ACTION_START : LAST_ACTION_END]
    card_oh = last_action_vec[NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS + NUM_RANKS + HAND_SIZE:
                              NUM_PLAYERS + 4 + NUM_PLAYERS + NUM_COLORS + NUM_RANKS + HAND_SIZE + HAND_SIZE]
    return np.argmax(card_oh)
    
def count_cards_in_observed_hands(obs, observer):
    count_vector = np.zeros(BITS_PER_CARD, dtype = 'int')
    hands = obs['player_observations'][observer]['observed_hands'][1:]
    for hand in hands:
        for card in hand:
            card_id = card_index(card)
            count_vector[card_id] +=1
    return count_vector



def count_cards_in_observed_hands_vec(obs_vector):
    count_vector = np.zeros(BITS_PER_CARD, dtype = 'int')
    for i in range(0, HAND_SIZE):
        count_vector += obs_vector[OBSERVED_HANDS_START + i * BITS_PER_CARD : (i + 1) * BITS_PER_CARD]
    return count_vector

def count_cards_in_fireworks_vec(obs_vec):
    fireworks_vec = obs_vec[FIREWORKS_START : FIREWORKS_END]
    return fireworks_vec

def count_cards_in_discard_vec(obs_vec):
    count_vec = np.zeros(BITS_PER_CARD, dtype = 'int')
    discard_vec = obs_vec[DISCARD_START : DISCARD_END]
    index = 0
    for color_ind in range(NUM_COLORS):
        for rank in range(NUM_RANKS):
            card_ind = color_ind * NUM_RANKS + rank
            card_count = INITIAL_CARD_COUNT[card_ind]
            count_vec[card_ind] += np.sum(discard_vec[index : index + card_count]) 
            index += card_count
    return count_vec


def count_all_observed_cards(obs, observer):
    return (count_cards_in_observed_hands(obs, observer) + count_cards_in_fireworks(obs, observer)
            + count_cards_in_discard(obs, observer))

def count_unknown_cards(obs, observer):
    cards_observed = count_all_observed_cards(obs, observer)
    
def count_all_observed_cards_vec(obs_vec):
    return (count_cards_in_observed_hands_vec(obs_vec,) + count_cards_in_fireworks_vec(obs_vec)
            + count_cards_in_discard_vec(obs_vec,))

def count_unknown_cards_vec(obs_vec):
    cards_observed = count_all_observed_cards_vec(obs_vec)
    return (INITIAL_CARD_COUNT - cards_observed)
       
def get_action_type(action):
    return HINT_MASK[action]


    