#
#
"""First attempt on building the rule based agent"""

#from hanabi_learning_environment import rl_env
from hanabi_learning_environment.rl_env import Agent
import gin.tf
import operator
import numpy as np
from pyhanabi import HanabiMoveType as move_type

D = False # set to True if you want to print out useful info

@gin.configurable
class RuleBasedAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, players, *args, **kwargs):
        """Initialize the agent."""
        # set number of players
        self.players = players

        if self.players < 4:
            num_cards = 5
        else:
            num_cards = 4

        self.rank_hinted_but_no_play = [0] * players

        for i in range(len(self.rank_hinted_but_no_play)):
            self.rank_hinted_but_no_play[i] = [False] * num_cards

        # Extract max info tokens or set default to 8.
        self.max_information_tokens = 8 #config.get('information_tokens', 8)

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        if card['rank'] is None:
            return False
        else:
            return card['rank'] == fireworks[card['color']]

    def check_if_not_playable_hint(self, observation):

        # check if rank hint was hinted for discard or not,
        last_moves = observation['pyhanabi'].last_moves()

        own_cards_knowledge = observation['pyhanabi'].card_knowledge()[0]

        for last_move in last_moves:
            player_idx = last_move.player()
            move = last_move.move()
            target_offset = move.target_offset()
            # check if hint was from player
            # print("revealed in last move", last_move.card_info_revealed())
            # print("was 0 card index in move?:", 0 in last_move.card_info_revealed())
            # print("offset: ", move.target_offset())
            # print("Is rank same as card checking?: ", move.rank() == own_card_knowledge.rank())
            if move.type() == move_type.REVEAL_RANK and move.target_offset() == 1 \
                    and 0 in last_move.card_info_revealed() and \
                    move.rank() is not None:
                player_target_idx = (player_idx + 1) % (self.players - 1)
                # hint is from left partner, not useful hint though (just hint to free tokens)')
                for card_idx in last_move.card_info_revealed():

                    self.rank_hinted_but_no_play[player_target_idx][card_idx] = True
                break
            elif move.type() == move_type.DISCARD or move.type() == move_type.PLAY:
                self.rank_hinted_but_no_play[player_idx].pop(move.card_index())
                self.rank_hinted_but_no_play[player_idx].append(False)
        if D:
            print("Current state of", self.rank_hinted_but_no_play[0])

    def maybe_play_lowest_playable_card(self, observation):
        """
        The Bot checks if previously a card has been hinted to him,
        :param observation:
        :return:
        """

        own_card_knowledge = observation['pyhanabi'].card_knowledge()[0]

        if D:
            print('Own card knowledge', observation['pyhanabi'].card_knowledge()[0])
            print()
        for index, own_card_know in enumerate(own_card_knowledge):
            if own_card_know.color() is not None:
                if D:
                    print("Will play from color hint card: ", own_card_know, "at index: ", index)
                self.rank_hinted_but_no_play[0].pop(index)
                self.rank_hinted_but_no_play[0].append(False)
                return {
                    'action_type': 'PLAY',
                    'card_index': index
                }
        for index, own_card_know in enumerate(own_card_knowledge):
            if own_card_know.rank() is not None and \
                    not self.rank_hinted_but_no_play[0][index]:
                self.rank_hinted_but_no_play[0].pop(index)
                self.rank_hinted_but_no_play[0].append(False)
                if D:
                    print("Will play from value hint card: ", own_card_know, "at index: ", index)
                return {
                    'action_type': 'PLAY',
                    'card_index': index
                }
            else:
                if D:
                    print("Not enough info for Card ", own_card_know, " to play at index: ", index)

    def maybe_give_helpful_hint(self, observation):

        if observation['information_tokens'] is 0:
            return None

        fireworks = observation['fireworks']

        best_so_far = 0
        player_to_hint = -1
        color_to_hint = -1
        value_to_hint = -1

        # for player_offset in range(1, observation['num_players']):
        #    print('Cards from partner {}'.format(player_offset))
        #    print(observation['observed_hands'][player_offset])
        for player_offset in range(1, observation['num_players']):
            player_hand = observation['observed_hands'][player_offset]
            # player_hints = observation['card_knowledge'][player_offset]
            player_knowledge = observation['pyhanabi'].card_knowledge()[player_offset]

            if D:
                print()
                print("Cards from partner are: ")
                print(player_hand)
                print()
                print("Knowledge partner know over his card: ")
                print(player_knowledge)
                print()

            # Check if the card in the hand of the opponent is playable.

            card_is_really_playable = [False, False, False, False, False]
            playable_colors = []
            playable_ranks = []
            for index, (card, hint) in enumerate(zip(player_hand, player_knowledge)):
                if self.playable_card(card, fireworks):
                    if D:
                        print("Card ", card, 'at index:', index, "is playable")
                    card_is_really_playable[index] = True
                    if card['color'] not in playable_colors:
                        playable_colors.append(card['color'])
                    if card['rank'] not in playable_ranks:
                        playable_ranks.append(card['rank'])

            '''Can we construct a color hint 
            that gives our partner information
            about unknown - playable cards, 
            without also including any unplayable cards?'''

            # go through playable colors
            for color in playable_colors:
                if D:
                    print('playable color is: ', color)
                information_content = 0
                missinformative = False
                for index, (card, knowledge) in enumerate(zip(player_hand, player_knowledge)):
                    if card['color'] is not color:
                        continue
                    if self.playable_card(card, fireworks) and \
                            knowledge.color() is None:
                        if D:
                            print('Hint for color {} is informative'.format(color))
                        information_content += 1
                    elif not self.playable_card(card, fireworks):
                        missinformative = True
                        break
                if missinformative:
                    continue
                if information_content > best_so_far:
                    best_so_far = information_content
                    color_to_hint = color
                    value_to_hint = -1
                    player_to_hint = player_offset
                    if D:
                        print()
                        print("Best hint at the moment: color{} to player{}".format(color_to_hint, player_to_hint))
                        print()

            # go through playable ranks
            for rank in playable_ranks:
                information_content = 0
                missinformative = False
                for index, (card, knowledge) in enumerate(zip(player_hand, player_knowledge)):
                    if card['rank'] is not rank:
                        continue
                    if self.playable_card(card, fireworks) and \
                            (knowledge.rank() is None or self.rank_hinted_but_no_play[player_offset][index]):
                        information_content += 1
                    elif not self.playable_card(card, fireworks):
                        missinformative = True
                        break
                if missinformative:
                    continue
                if information_content > best_so_far:
                    best_so_far = information_content
                    color_to_hint = None
                    value_to_hint = rank
                    player_to_hint = player_offset

        # went through all players, now check
        if best_so_far is 0:
            return None
        #
        elif color_to_hint is not None:
            return {
                'action_type': 'REVEAL_COLOR',
                'color': color_to_hint,
                'target_offset': player_to_hint
            }
        elif value_to_hint is not -1:
            return {
                'action_type': 'REVEAL_RANK',
                'rank': value_to_hint,
                'target_offset': player_to_hint
            }
        else:
            return None

    def act(self, observation):
        """
        Act by making a move, depending on the observations.
        :param observation: Dictionary containing all information over the hanabi game, from the view
        of the players
        :return: Returns a dictionary, describing the action
        """

        # check if in previous round, the left partner (index = num_players-1) has given us a VALUE-HINT,
        # in which our latest card (index = 0) was hinted. These cards are not necessarily playable, therefore they need
        # to be marked
        self.check_if_not_playable_hint(observation)

        # check if it is our turn, only give action then
        if observation['current_player_offset'] != 0:
            return None

        # If I have a playable card, play it.
        action = self.maybe_play_lowest_playable_card(observation)
        if action is not None:
            return action

        # Otherwise, if someone else has an unknown-playable card, hint it.
        action = self.maybe_give_helpful_hint(observation)
        if action is not None:
            return action

        # We couldn't find a good hint to give, or we are out of hint-stones.
        # We wil discard a card, if possible.
        # Otherwise just hint the next play
        isDiscardingAllowed = False
        for legal_moves in observation['legal_moves']:
            if legal_moves['action_type'] is 'DISCARD':
                isDiscardingAllowed = True
                break
        if not isDiscardingAllowed:
            # assume next player in turn in player on right
            hand_on_right = observation['observed_hands'][1]

            # Only hinting because no better option,
            return {
                'action_type': 'REVEAL_RANK',
                'rank': hand_on_right[0]['rank'],
                'target_offset': 1
            }
        else:
            # Discard our oldest card
            self.rank_hinted_but_no_play[0].pop(0)
            self.rank_hinted_but_no_play[0].append(False)
            return {
                'action_type': 'DISCARD',
                'card_index': 0
            }

    def act_train(self, observation):

        action = self.act(observation)

        #get the corresponding action in from legal moves
        legal_move_idx = -1

        for idx, legal_action in enumerate(observation['legal_moves']):

            if operator.eq(action, legal_action):
                legal_move_idx = idx
                break
        return np.int_(observation['legal_moves_as_int'][legal_move_idx])



    @staticmethod
    def is_rl_agent():
        return False
