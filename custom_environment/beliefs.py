import numpy as np
from custom_environment.utils import REVEAL_RANK, REVEAL_COLOR, PLAY, DISCARD, COLOR_CHAR
    
class Belief:
    def __init__(self, game, num):
        
        self.num = num
        self.game = game
        self.hand_size = game.hand_size()
        self.num_colors = game.num_colors()
        self.num_ranks = game.num_ranks()
        self.bits_per_card = self.num_colors * self.num_ranks
        self.initial_card_count = np.asarray([game.num_cards(c, r) for c in range(self.num_colors)
                                        for r in range(self.num_ranks)], dtype = 'int').reshape((-1, 1))
        
        
        
    def encode_cards(self, card_list, dtype = 'int'):
        res = np.zeros(self.bits_per_card, dtype = dtype)
        for c in card_list:
            res[c.color() * self.num_ranks + c.rank()] += 1
        return res
    
    @property
    def card_count(self,):
        return self.initial_card_count - self.known_cards
    
    @property
    def masked_card_count(self):
        return self.card_masks * self.card_count
    
    
    def reset(self, state):
        self.state = state
        self.own_hand = (self.state.player_hands()[self.num])
        self.known_cards = self.encode_cards(self.state.player_hands()[1 - self.num]).reshape((-1, 1))
        self.card_masks = np.ones((self.hand_size, self.bits_per_card, 1))
        
        
        
    def observe_action(self, action):
        
        if action.type() == PLAY or action.type() == DISCARD:
            if len(self.state.player_hands()[1 - self.num])  == self.hand_size:
                new_card = self.state.player_hands()[1 - self.num][self.hand_size - 1]
                new_card_ind = new_card.color() * self.num_ranks + new_card.rank()
                self.known_cards[new_card_ind] += 1

        elif action.type() == REVEAL_RANK or action.type() == REVEAL_COLOR:
            hint_mask = np.zeros((self.bits_per_card, 1), dtype = 'int')
            if action.type() == REVEAL_RANK:
                hint_mask[[self.num_ranks * i + action.rank() for i in range(self.hand_size)]] = 1
            else:
                hint_mask[action.color() * self.num_ranks : (action.color() + 1) * self.num_ranks] = 1
            #print('Hint mask:', hint_mask)
            card_masks_factor = np.ones((self.hand_size, self.bits_per_card, 1))
            for i, c in enumerate(self.own_hand):
                if c.rank() == action.rank() or c.color() == action.color():
                    card_masks_factor[i] = hint_mask
                else:
                    card_masks_factor[i] = 1 - hint_mask
            self.card_masks *= card_masks_factor
        else:
            print('Unrecognized action type ', action.type)
        
    def make_action(self, action):
        
        if action.type() == PLAY or action.type() == DISCARD:
            index = action.card_index()
            played_card = self.own_hand[index]
            played_card_ind = played_card.color() * self.num_ranks + played_card.rank()
            if index == 0:
                self.card_masks[0], self.card_masks[1] = (np.array(self.card_masks[1]),
                                                          np.ones((self.bits_per_card, 1)))
            elif index == 1:
                self.card_masks[1] = np.ones((self.bits_per_card, 1))
                
            self.own_hand = self.state.player_hands()[self.num]
            if len(self.own_hand)  == self.hand_size:
                new_card = self.own_hand[self.hand_size - 1]
                self.known_cards[played_card_ind] += 1
            
    def update_hands_info(self):
        self.hands_count = (np.dot(self.card_count, self.card_count.T) - 
                            np.identity(10) * self.card_count)
        self.hands_mask = self.card_masks[0] * self.card_masks[1].T
        
    @property
    def first_card_prior_probs(self):
        probs = (self.card_count * self.card_masks[0]/ np.sum(self.card_count * self.card_masks[0]))
        return probs
    @property
    def second_card_prior_probs(self):
        probs = (self.card_count * self.card_masks[1]/ np.sum(self.card_count * self.card_masks[1]))
        return probs
    @property
    def second_card_conditioned_probs(self):
        fixed_count = (self.card_count.T - np.identity(10)) * self.card_masks[1].T
        fixed_count[fixed_count <= 0] = 0
        denom = (np.sum(fixed_count, 1)[:, np.newaxis])
        denom[denom == 0] = 1e-5
        return fixed_count / denom
    
    @property
    def first_card_conditioned_probs(self):
        fixed_count = (self.card_count - np.identity(10)) * self.card_masks[0]
        fixed_count[fixed_count <= 0] = 0
        denom = (np.sum(fixed_count, 0))
        denom[denom == 0] = 1e-5
        return fixed_count / denom
    
    def compute_hands_probs(self):
        self.update_hands_info()
        masked_hands_count = np.array(self.hands_count)
        masked_hands_count[self.hands_mask == 0] = 0
        return masked_hands_count / np.sum(masked_hands_count)
    
    def compute_all_probs(self):
        hand_probs = self.compute_hands_probs()
        first_prior = self.first_card_prior_probs
        second_prior = self.second_card_prior_probs
        first_cond = self.first_card_conditioned_probs
        second_cond = self.second_card_conditioned_probs
        return {'hand_probs' : hand_probs,
                'first_prior' : first_prior,
                'second_prior' : second_prior,
                'first_cond' : first_cond,
                'second_cond' : second_cond}
        