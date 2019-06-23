from rl_env import Agent


# called from rl_env_custom.py which we can basically copy from rl_env_example.py
class CustomAgent(Agent):

    """ Agent that implements rules for playing Hanabi.
    The rules are based on conventions taken from https://github.com/Zamiell/hanabi-conventions """

    def __init__(self, config, *args, **kwargs):
        """ Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8
        # TODO: add inits based on strategies
        self.max_information_tokens = config.get('information_tokens', 8)
        # keep track of clue history for delayed plays
        self.prior_clues = list()
        # list of indices that keeps track of cards that are clued directly
        self.focused = list()
        # list of indices that keeps track of cards that are clued, convenience method
        self.clued = list()

    @staticmethod
    def playable_card(card, fireworks):
        """ A card is playable if it can be placed on the fireworks pile."""
        return card['rank'] == fireworks[card['color']]

    @staticmethod
    def get_critical_cards(discard_pile, observed_hands):
        """
        A card is critical, if the remaining copies have been discarded.
        """
        # return critical_cards as list of (color, rank, player_offset) tuples
        critical_cards = []

        # count occurences of discarded cards
        count_discarded = dict()  # dict with keys like 'W0' for white 1 or 'G4' for green 5
        for d in discard_pile:
            key = (d['color'], d['rank'])
            count_discarded[key] = count_discarded.get(key, 0) + 1

        # iterate all cards observable by the current player
        for hand in observed_hands:
            for card in hand:
                # detect critical cards:
                for color, rank in count_discarded:
                    count = count_discarded[(color, rank)]
                    # 1 is critical if 2 of the same kind are discarded, 5 is always critical
                    if count >= (1 - (rank - 1)//3):  # card is critical
                        if card['color'] == color and card['rank'] == rank:
                            critical_cards.append(card)
        return critical_cards

    # @staticmethod
    # def get_unique_twos(observed_hands, num_players):
    #     """
    #     A 2 is unique, if only one copy of it is currently visible.
    #     """
    #     # return unique 2s as dict with key color and val player_offset
    #     unique_twos = dict()
    #     for player_offset in range(1, num_players):
    #         hand = observed_hands[player_offset]
    #         for card in hand:
    #             if card['rank'] == 2:  # find unique 2s (only one copy currently visible)
    #                 if card['color'] in unique_twos:
    #                     del unique_twos[card['color']]
    #                 else:
    #                     unique_twos[card['color']] = player_offset
    #
    # @staticmethod
    # def get_unclued_fives(observed_hands):
    #     # return unclued 5s
    #     pass
    #

    @staticmethod
    def is_critical(card, discard_pile):
        card_is_critical = False

        # count occurences of discarded cards
        count_discarded = dict()  # dict with keys like 'W0' for white 1 or 'G4' for green 5
        for card in discard_pile:
            key = (card['color'], card['rank'])
            count_discarded[key] = count_discarded.get(key, 0) + 1

        # if card matches discarded card, check if there are no remaining copies and if so, return True
        for color, rank in count_discarded:
            count = count_discarded[(color, rank)]
            # 1 is critical iff 2 of the same kind are discarded
            # 2,3,4 are critical iff 1 of the same kind are discarded
            # 5 is criticial iff 0 of the same kind are discarded
            if count >= (1 - (rank - 1) // 3):  # card is critical
                if card['color'] == color and card['rank'] == rank:
                    card_is_critical = True

        return card_is_critical

    @staticmethod
    def get_chops(observation, num_players):
        # return list of (chop_card, player_offset) tuples
        # the chop card is the unclued card with lowest index
        chops = []
        for player_offset in range(1, num_players):
            player_hand = observation['observed_hands'][player_offset]
            player_hints = observation['card_knowledge'][player_offset]
            for card, hint in zip(player_hand, player_hints):
                if hint['color'] is None and hint['rank'] is None:
                    # chop of current player is the first unclued card
                    chops.append(tuple(card, player_offset))
                    break

        # if all cards are clued, there is no chop
        return chops

    @staticmethod
    def is_chop(card_idx, hand_knowledge):
        """Determines for a given card index, if it corresponds to the chop card given hand_knowledge."""

        # get chop card to compare it
        for index, hint in enumerate(hand_knowledge):
            if hint['color'] is None and hint['rank'] is None:
                # chop of current player is the first unclued card
                chop_idx = index
                return chop_idx == card_idx

        # if all cards are clued, there is no chop
        return False

    def get_focused_other(self, hand, clue, clue_type, hand_knowledge):
        """ Returns the card that is focused by the given clue. A clue can only focus one of the touched cards.
        If the clue only touches one card, this card is returned trivially.
        If it touches multiple cards, the chop will be the focused card, if touched, else the newest card is focused."""

        # A clue is either one of 'R', 'Y', 'G', 'W', 'B' or one of 0,1,2,3,4
        touched = [card for card in hand if card[clue_type] == clue]
        assert len(touched) > 0  # Make sure a clue touches at least one card

        # if exactly one card is touched, then it trivially is the focus of the clue
        if len(touched) == 1:
            return touched[0]

        # when multiple cards a touched by the given clue
        # focus the chop, if it is touched
        for touched_card in touched:
            if self.is_chop(hand.index(touched_card), hand_knowledge):
                return touched_card

        # else the newest card is focused by the clue
        return touched[-1]

    def has_playable_card(self, hand, fireworks):
        """Returns True, playable_cards if hand contains cards that """
        playable_cards = None
        has_playable = False
        for card in hand:
            if self.playable_card(card, fireworks):
                playable_cards.append(card)

        return has_playable, playable_cards

    def get_clued_self(self):

        pass

    def act(self, observation):

        """ Act based on observation.
        1. If a player needs to be given a save clue, only give it, if no play clue can be given to the same player
        2. Only give save clues on chop cards, because these are being discarded in case of DISCARD action
        3. If a card is focused by a clue, that is not a save clue, it must be a (delayed) play clue!
        4. Test this: A play clue is delayed, iff it is a finesse move.
        """
        # check if we got a clue in the past round
        # get new clued cards
        # update touched
        # return action
        action = None
        action_type = 'DISCARD'

        # Only perform an action when its the agents turn
        if observation['current_player_offset'] != 0:
            return action

        # get number of players including self
        num_players = observation['num_players']

        # get chop card for each teammate
        chops = self.get_chops(observation, num_players)

        # check if one of the chop cards needs to be saved
        for (chop_card, player_offset) in chops:
            # note that a chop by definition has no clue on it
            # priority is given first by player_offset then by
            # 1) is_critical ? 2) is a five? 3) is_unique_two?
            if self.is_critical(chop_card, observation['discard_pile']):
                # set action, s.t. it touches fewest additional cards
                # if possible, give play clue over save clue
                if player_offset == 1:  # TODO: or agent has no playable card, then he might as well give the clue
                    # player has to give the save hint with either REVEAL_COLOR or REVEAL_RANK
                    pass
            pass

        if action_type == 'DISCARD' or action_type == 'PLAY':
            # shift by one or remove corresponding card
            self.prior_clues = None





    """BASICALLY WE NEED LOGS AND AS SUCH THE RL_ENV IS TO HIGHLEVEL. We WILL HAVE TO MIGRATE TO THE GAME EXAMPLE"""



    # In case we want to change strategies after n rounds of playing.
    def reset(self, config):
        pass

