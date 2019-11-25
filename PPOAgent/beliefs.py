import numpy as np
from .gamesettings import *
from CardCounting import *
def concatenate_two_arrays_per_row(A, B):
    m1,n1 = A.shape
    m2,n2 = B.shape

    out = np.zeros((m1,m2,n1+n2),dtype=A.dtype)
    out[:,:,:n1] = A[:,None,:]
    out[:,:,n1:] = B
    return out.reshape(m1*m2,-1)

def concatenate_per_row(list_of_arrays):
    shapes = []
    for arr in list_of_arrays:
        shapes.append(arr.shape)
    shapes = np.array(shapes)
    out_shape = shapes[:, 0].tolist() + [np.sum(shapes, 0)[1]]
    out = np.zeros(out_shape, dtype = list_of_arrays[0].dtype)
    current_offset = 0
    fakesaxises = []
    for arr, n in zip(list_of_arrays, shapes[:, 1]):
        newshape = [arr.shape[0]] + fakesaxises + [arr.shape[1]]
        out[:,..., current_offset : current_offset + n] = arr.reshape(newshape)
        fakesaxises.append(1)
        current_offset += n
    return out.reshape(np.product(shapes, 0)[0], -1)



class Belief:
    
    def __init__(self):
        self.all_hands = []
        for c1 in range(BITS_PER_CARD):
            for c2 in range(BITS_PER_CARD):
                hand = np.zeros(2 * BITS_PER_CARD, dtype = 'int')
                hand[[c1, BITS_PER_CARD + c2]] = 1
                self.all_hands.append(hand)
        self.reset()
        
    def reset(self):
        #print('reset')
        self.masks = [np.ones(BITS_PER_CARD, dtype = 'int') for _ in range(HAND_SIZE)]
        self.cards_count = INITIAL_CARD_COUNT
        self.current_hand_size = HAND_SIZE
        self.update_pools()
        
    def update_pools(self):
        self.pools = [(self.cards_count * m) for m in self.masks]
        self.pool_sizes = [np.sum(pool) for pool in self.pools]
        self.shared_pool = self.cards_count * np.prod(self.masks, 0)
        self.shared_pool_size = np.sum(self.shared_pool)
        
    def process_observation(self, obs_vec, update = True):
        old = self.cards_count
        self.cards_count = count_unknown_cards_vec(obs_vec)
        #print(old, '--->',self.cards_count)
        last_action = get_last_action_type(obs_vec)
        if last_action in [3, 4]:
            # if other player gave hint
            hinted_cards = get_cards_hinted_vec(obs_vec)
            hint_mask = get_mask_from_hint(obs_vec)
            for i in range(HAND_SIZE):
                self.masks[i] = self.masks[i] * (hint_mask if hinted_cards[i] else 1 - hint_mask)
        self.update_pools()

    def process_played_card(self, obs_vec, offset = 0):
        affected_card = get_card_played(obs_vec)
        draw_new_card = 1 - obs_vec[OBSERVED_HANDS_END + offset]
        self.masks.pop(affected_card)
        if not draw_new_card:
            
            self.current_hand_size -= 1
            #print('became sh', self.current_hand_size )
            self.masks.append(np.zeros(BITS_PER_CARD, dtype = 'int'))
        else:
            self.masks.append(np.ones(BITS_PER_CARD, dtype = 'int'))
        self.update_pools()
        
    def compute_first_card_prob(self, card1):
        card_plausible = self.masks[0][card1]
        if not card_plausible:
            return 0
        nom = self.cards_count[card1]
        if nom <= 0:
            return 0
        denom = self.pool_sizes[0]
        if denom == 0:
            print('cpf', card1, nom, denom, self.cards_count, self.masks)
        return nom / denom
    
    def compute_second_card_prob(self, card2):
        card_plausible = self.masks[1][card2]
        if not card_plausible:
            return 0
        nom = self.cards_count[card2]
        if nom <= 0:
            return 0
        denom = self.pool_sizes[1]
        if denom == 0:
            print('cps', card2, nom, denom, self.cards_count, self.masks)
        return nom / denom
        
    def compute_first_by_second(self, card1, card2):
        plausible = self.masks[0][card1]
        if not plausible:
            return 0
        card_in_pool = min(self.pools[0][card2], 1)
        nom = self.cards_count[card1] - int(card1 == card2)
        if nom <= 0:
            return 0
        denom = self.pool_sizes[0] - card_in_pool
        if denom == 0:
            print('cpfbs', card1, card2, nom, denom, self.cards_count, self.masks)
        return nom / denom
    
    def compute_second_by_first(self, card1, card2):
        plausible = self.masks[1][card2]
        if not plausible:
            return 0
        card_in_pool = min(self.pools[1][card1], 1)
        nom = self.cards_count[card2] - int(card1 == card2)
        if nom <= 0:
            return 0
        denom = self.pool_sizes[1] - card_in_pool
        if denom == 0:
            print('cpsbf', card1, card2, nom, denom, self.cards_count, self.masks)
        return nom / denom
    
    def compute_hand_prob(self, card1, card2):
        card1_plausible = self.masks[0][card1]
        card2_plausible = self.masks[1][card2]
        if not card1_plausible or not card2_plausible:
            return 0
        nom = self.cards_count[card1] * (self.cards_count[card2] - int(card1 == card2))
        denom = self.pool_sizes[0] * self.pool_sizes[1] - self.shared_pool_size
        if denom == 0:
            print('cph', 'hs:', self.current_hand_size, 'hand', card1, card2, nom, denom, self.cards_count, self.masks)
        return nom / denom
    
    def compute_possible_hands_with_probs(self):
        probs = []
        first_card_mask = [[] for _ in range(BITS_PER_CARD)]
        second_card_mask = [[] for _ in range(BITS_PER_CARD)]
        index = 0
        for card1 in range(BITS_PER_CARD):
            for card2 in range(BITS_PER_CARD):
                hand_prob = self.compute_hand_prob(card1, card2)
                probs.append(hand_prob)

                first_card_mask[card1].append(index)
                second_card_mask[card2].append(index)
                index += 1
        return np.array(probs), np.array(first_card_mask), np.array(second_card_mask)
    
    def compute_hand_beliefs(self, MM_output):
        #print('hand size', self.current_hand_size)
        #print('count', self.cards_count)
        #print('masks', self.masks)
        hands = self.all_hands
        hands_probs, fc_mask, sc_mask = self.compute_possible_hands_with_probs()  
        
        first_card_prior_probs = [self.compute_first_card_prob(c) for c in range(BITS_PER_CARD)]
        #print('FC_prior', first_card_prior_probs)
        second_card_prior_probs = [self.compute_second_card_prob(c) for c in range(BITS_PER_CARD)]
        if self.current_hand_size == 1:
            return np.concatenate([first_card_prior_probs, second_card_prior_probs], 0)
        
        #print('SC_prior', second_card_prior_probs)
        first_card_conditioned_by_second = np.array([[self.compute_first_by_second(c1, c2)
                                                     for c2 in range(BITS_PER_CARD)] 
                                                     for c1 in range(BITS_PER_CARD)])
        #print('FC_cond', first_card_conditioned_by_second)
        second_card_conditioned_by_first = np.array([[self.compute_second_by_first(c1, c2)
                                                     for c2 in range(BITS_PER_CARD)] 
                                                     for c1 in range(BITS_PER_CARD)])
        #print('SC_cond', second_card_conditioned_by_first)

        action_conditioned_by_first_card = np.array([np.sum(MM_output[fc_mask[c1]] *\
                                                            second_card_conditioned_by_first[c1])     
                                                     for c1 in range(BITS_PER_CARD)])

        action_conditioned_by_second_card = np.array([np.sum(MM_output[sc_mask[c2]] *\
                                                             first_card_conditioned_by_second[:, c2])    
                                                      for c2 in range(BITS_PER_CARD)])

        
        action_prior_prob = np.sum(MM_output * hands_probs)
        if action_prior_prob == 0:
            print(MM_output)
            print(hands_probs)
        first_card_probs = action_conditioned_by_first_card * first_card_prior_probs / action_prior_prob
        second_card_probs = action_conditioned_by_second_card * second_card_prior_probs / action_prior_prob

        return np.concatenate([first_card_probs, second_card_probs], 0)
    
        