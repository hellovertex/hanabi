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

    @staticmethod
    def playable_card(card, fireworks):
        """ A card is playable if it can be placed on the fireworks pile."""
        return card['rank'] == fireworks[card['color']]

    def act(self, observation):

        """ Act based on observation."""
        if observation['current_player_offset'] != 0:
            return None  # Only acts on its turn
        DISCARD_FLAG = False

        # ------- PRELIMINARIES ------- #
        # Rule 1: Oldest = Right-most // Newest = Left-most
        # => Oldest card is at lowest index and conversely, the newest card is at the highest index

        # ############################# #
        #          The Basics           #
        # ############################# #

        # -------- The Chop ----------- #
        """ 
        # Rule 2: chop = Right-most unclued
        # => chop = unclued with lowest index
        """
        # Set chop card
        chop = None
        for card_index, hint in enumerate(observation['card_knowledge'][0]):
            if hint['color'] is None and hint['rank'] is None:
                chop = card_index
                break

        # If discard, then discard chop
        if DISCARD_FLAG:
            return {'action_type': 'DISCARD', 'card_index': chop}

        # ---- Single Card Focus ----- #
        """
        # Specify WHAT a clue means
        # If two or more cards are touched, the clue is focused on a only a single card
        # => Play Clue or Save Clue ONLY on the focused card, i.e. NOTHING is implied on the unfocused cards
        # focus is always on the card that did not have any clues alredy [BRAND NEW CARD INTRODUCED]
        # if there are multiple UNCLUED cards, 
        # if the chop is new, set the focus on the chop, else set it to the leftmost card
        """

        # set focused
        # get touched cards
        # check if card that one wants to clue, has already been played (on firework)
        # or is already clued (possibly in someone elses hand)

        # observation['observed_hands'] = List[List[Dict[str]]]
        # 'rank' : -1 to 5, 'color': None or in 'RYGWB'

        # observation['card_knowledge'] = List[List[Dict[str]]]
        # # 'rank' : -1 to 5, 'color': None or in 'RYGWB'


        # utils
        fireworks = observation['fireworks']
        discarded = observation['discard_pile']
        observed_hands = observation['observed_hands']
        card_knowledge = observation['card_knowledge']
        num_players = observation['num_players']
        tokens_remaining = observation['information_tokens']

        # count occurences of discarded cards
        count_discarded = dict()  # dict with keys like 'W0' for white 1 or 'G4' for green 5
        for d in discarded:
            key = (d['color'], d['rank'])
            count_discarded[key] = count_discarded.get(key, 0) + 1

        # When information token is available, look for urgent hints
        if tokens_remaining > 0:
            unique_twos = dict()
            # Iterate the teammates cards and their knowledge of them
            for player_offset in range(1, num_players):
                player_hand = observed_hands[player_offset]
                player_hints = card_knowledge[player_offset]

                for card, hint in zip(player_hand, player_hints):

                    # search other players hands for cards that need to be saved
                    if card['rank'] == 4:  # find unclued 5s:
                        if hint['color'] is None and hint['rank'] is None:
                            # return save clue for 5
                            return {
                                'action_type': 'REVEAL_RANK',
                                'rank': card['rank'],
                                'target_offset': player_offset
                            }
                    if card['rank'] == 2:  # find unique 2s (only one copy currently visible)
                        if card['color'] in unique_twos:
                            del unique_twos[card['color']]
                        else:
                            unique_twos[card['color']] = (card, player_offset)
                    #
                    # detect critical cards:
                    for color, rank in count_discarded:
                        count = count_discarded[(color, rank)]
                        if count >= 2 - (rank + 3) // 4:  # card is critical
                            if card['color'] == color and card['rank'] == rank:
                                # return save clue for critical card
                                return {
                                    'action_type': 'REVEAL_COLOR',
                                    'color': card['color'],
                                    'target_offset': player_offset
                                }

            # hint unique two to player_offset
            if len(unique_twos) > 0:
                for color, (card, player_offset) in unique_twos.items():
                    return {
                        'action_type': 'REVEAL_RANK',
                        'rank': card['rank'],
                        'target_offset': player_offset
                    }










    # In case we want to change strategies after n rounds of playing.
    def reset(self, config):
        pass

