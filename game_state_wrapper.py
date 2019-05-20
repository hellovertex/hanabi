import ast
from typing import Optional, List, Set
import copy


class GameStateWrapper:

    def __init__(self, num_players, agent_name):
        """
        # ################################################ #
        # -------------------- CONFIG -------------------- #
        # ################################################ #
        """
        self.players = None  # list of names of players currently ingame
        self.agent_name = agent_name  # used to identify absolute position on table
        self.num_players = num_players  # number of players ingame
        self.cards_per_hand = 4 if num_players > 3 else 5  # deal 5 cards when playing with 2 or 3 ppl
        self.deck_size = None  # number of remaining cards in the deck
        self.max_life_tokens = 3
        self.max_info_tokens = 8
        self.life_tokens = 3  # todo get from config
        self.information_tokens = 8  # todo get from config
        self.deck_size = 50  # todo get from config

        """
        # ################################################ #
        # ------- Observed Cards and Card knowledge ------ #
        # ################################################ #
        """
        # list of all players hands as _seen_ by calling agent [excluding clues]
        self.hand_list = list()  # is refreshed in self.update() on each notify message

        # list of clues given
        self.clues = list()  # is refreshed in self.update() on each notify message

        # unfortunately, server references clue cards not by index but by an id between 0 and deck size,
        # so we need to store card_numbers to map the card ids to indices
        self.card_numbers = list()


        """
        # ################################################ #
        # ----------------- GAME STATS ------------------- #
        # ################################################ #
        """
        # is refreshed in self.update() on each notify message
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}

        # list of discarded cards as returned by self.card(suit, rank)
        self.discard_pile = list()

        # actually not contained in the returned dict of the
        # rl_env.HanabiEnvobservation._extract_from_dict method, but we need a history so we add this here.
        # Similarly, it can be added by appending obs_dict['last_moves'] = observation.last_moves() in said method.
        self.last_moves = list()
        self.agents_turn = False

    def init_players(self, notify_msg: str):
        """ Sets self.players to a list of the players currently ingame and creates empty hands """
        self.reset()
        player_dict = ast.literal_eval(notify_msg.split('init ')[1].replace('false', 'False').replace('list',
                                                                                                      'List').replace('true', 'True'))
        self.players = player_dict['names']
        self.num_players = len(self.players)
        self.hand_list = [list() for _ in range(self.num_players)]
        self.card_numbers = [list() for _ in range(self.num_players)]
        self.clues = [list() for _ in range(self.num_players)]

        return

    def deal_cards(self, notify_msg):
        """ Initializes self.hand_list from server message 'notifyList [{"type":"draw","who":0,"rank":4,"suit":1,
        "order":0},...'"""

        # list of dictionaries storing the draws
        card_list = ast.literal_eval(notify_msg.split('notifyList ')[1].replace('false', 'False').replace('list',
                                                                                                          'List').replace('true', 'True'))

        for d in card_list:
            if d['type'] == 'draw':  # add card to hand of player with id d['who'] from left to right
                self.draw_card(d)
                # the new card has no clues on it when drawn
                self.clues[d['who']].append({'color': None, 'rank': None})

            if d['type'] == 'turn':
                # notifyList message also contains info on who goes first
                if d['who'] == self.players.index(self.agent_name):
                    self.agents_turn = True

        return

    def draw_card(self, d):
        """ Adds card to players hand and updates deck size. Then updates card references and clues."""
        # prepend drawn card to players hand
        self.hand_list[d['who']].insert(0, self.card(d['suit'], d['rank']))

        # decrease deck size counter
        self.deck_size -= 1

        # unfortunately, server references clued cards by absolute number and not its index on the hand
        # so we store the number too, to map it onto indices for playing and discarding
        self.card_numbers[d['who']].insert(0, d['order'])

    def discard(self, d):
        """
        Synchronizes references between handcards and clues.
        Need to reference by card_number, as we cannot access the card by rank and suit, if we discard own card,
        because its values are not known to us at the time of discarding
        """
        pid = d['which']['index']
        # Remove card number reference
        idx_card = self.card_numbers[pid].index(d['which']['order'])
        del self.card_numbers[pid][idx_card]

        # Remove card from hand
        del self.hand_list[pid][idx_card]

        # Remove card from clues
        self.clues[pid][idx_card] = {'color': None, 'rank': None}

        # Update discard pile
        self.discard_pile.append(self.card(d['which']['suit'], d['which']['rank']))

        return

    def update_state(self, notify_msg):
        """
        Updates Game State after server sends an action-notification.
        Notify message can contain 'turn', 'draw', 'play', 'clue', 'discard' as "type"-values
        """

        'Create dictionary from server message that contains actions'
        d = ast.literal_eval(notify_msg.split('notify ')[1].replace('false', 'False').replace('list',
                                                                                              'List').replace(
            'true', 'True'))
        if d['type'] in ['clue','discard', 'play']:
            print(d['type'])
            print(notify_msg)

        # DISCARD
        if d['type'] == 'discard':
            self.discard(d)
            # only recover info token when not discarding through failed play
            if 'failed' in d and d['failed'] is False:
                self.information_tokens += 1

        # DRAW - if player with pid draws a card, it is prepended to hand_list[pid]
        if d['type'] == 'draw':
            self.draw_card(d)

        # PLAY - remove played card from players hand and update fireworks/life tokens
        if d['type'] == 'play':
            # remove card
            self.discard(d)

            # update fireworks and life tokens eventually
            c = self.card(d['which']['suit'], d['which']['rank'])
            self.play(c)

        # CLUE - change players card_knowledge and remove an info-token
        if d['type'] == 'clue':
            self.update_clues(d)
            self.information_tokens -= 1  # validity has previously been checked by the server so were good with that

        # Set current player flag
        if d['type'] == 'turn':
            if d['who'] == self.players.index(self.agent_name):
                self.agents_turn = True
            else:
                self.agents_turn = False

        # On end of game, reset state
        if d['type'] == 'turn' and d['who'] == -1:
            pass

        # Add to history
        self.append_to_last_moves(d)

        return

    def update_clues(self, dict_clue):
        clue = dict_clue['clue']
        target = dict_clue['target']
        touched_cards = dict_clue['List']
        for c in touched_cards:
            idx_c = self.card_numbers[target].index(c)
            if clue['type'] == 0:
                # self.clues[target][idx_c]['rank'] = clue['value']
                self.clues[target][idx_c] = self.card(self.clues[target][idx_c]['color'], clue['value'])
            else:
                self.clues[target][idx_c] = self.card(clue['value'], self.clues[target][idx_c]['rank'])
        return

    def play(self, card):
        # on success, update fireworks
        if self.fireworks[card['color']] == card['rank']:
            self.fireworks[card['color']] += 1
        # on fail, remove a life token
        else:
            self.life_tokens -= 1
        return

    def append_to_last_moves(self, dict_action):
        """ looks like
        ############   DRAW   ##############
        notify {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
        ############   CLUE   ##############
        notify {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
        ############   PLAY   ##############
        notify {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
        ############   DISCARD   ##############
        notify {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
        """
        # set attributes for HanabiHistoryItem Mock object
        # we do this here, as the HistoryItemMock shall not have access to this classes attributes
        # but these are necessary to compute the information below. Its not the cleanest approach but readable at least

        move = self._get_pyhanabi_move_mock(dict_action)
        player = None
        scored = None
        information_token = None
        color = None
        rank = None
        card_info_revealed = None
        card_info_newly_revealed = None
        deal_to_player = None

        history_item_mock = HanabiHistoryItemMock(
            move=move,
            player=player,
            scored=scored,
            information_token=information_token,
            color=color,
            rank=rank,
            card_info_revealed=card_info_revealed,
            card_info_newly_revealed=card_info_newly_revealed,
            deal_to_player=deal_to_player
        )
        self.last_moves.append(history_item_mock)
        return

    def get_sorted_hand_list(self) -> List:
        """ Agent expects list of observations, always starting with his own cards. So we sort it here. """
        # moves self.cur_player hand to the front
        hand_list = copy.deepcopy(self.hand_list)
        hand_list.insert(0, hand_list.pop(hand_list.index(hand_list[self.players.index(self.agent_name)])))
        return hand_list

    def get_agent_observation(self):
        """ Returns state as perceived by the calling agent """
        observation = {
            'current_player': self.players.index(self.agent_name),
            'current_player_offset': 0,
            'life_tokens': self.life_tokens,
            'information_tokens': self.information_tokens,
            'num_players': self.num_players,
            'deck_size': self.deck_size,
            'fireworks': self.fireworks,
            'legal_moves': self.get_legal_moves(),
            'observed_hands': self.get_sorted_hand_list(),  # moves own hand to front
            'discard_pile': self.discard_pile,
            'card_knowledge': self.get_card_knowledge(),
            'vectorized': None,  # Currently not needed, we can implement it later on demand
            'last_moves': self.last_moves  # actually not contained in the returned dict of the
            # rl_env.HanabiEnvobservation._extract_from_dict method, but we need a history so we add this here.
            # Similarly, it can be added by appending obs_dict['last_moves'] = observation.last_moves() in said method.
        }
        return observation

    def card(self, suit: int, rank: int):
        """ Returns card format desired by agent. Rank values of None and -1 will be passed through."""

        if rank is not None:
            if rank > -1:  # return rank = -1 for an own unclued card
                rank -= 1  # server cards are not 0-indexed
        return {'color': self.convert_suit(suit), 'rank': rank}

    def next_player(self, offset):
        # make up for the fact, that we changed the order of the agents, s.t. self always is at first position
        idx = self.players.index(self.agent_name)

        if offset <= idx:
            return offset -1

        return offset

    @staticmethod
    def convert_suit(suit: int) -> Optional[str]:

        """
        Returns format desired by agent
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
        returns None if suit is None or -1
        """
        if suit == -1: return None
        if suit == 0: return 'B'
        if suit == 1: return 'G'
        if suit == 2: return 'Y'
        if suit == 3: return 'R'
        if suit == 4: return 'W'
        return None

    @staticmethod
    def convert_color(color: str) -> Optional[int]:
        """
        Returns format desired by server
            // 0 is blue
            // 1 is green
            // 2 is yellow
            // 3 is red
            // 4 is purple
        """
        if color is None: return -1
        if color == 'B': return 0
        if color == 'G': return 1
        if color == 'Y': return 2
        if color == 'R': return 3
        if color == 'W': return 4
        return -1

    def get_card_knowledge(self):
        """ Returns self.clues but formatted in a way desired by the agent"""
        card_knowledge = list()

        for hand in self.clues:
            h = list()
            for c in hand:
                # h.append(self.card(c['color'], c['rank']))
                h.append(c)
            card_knowledge.append(h)
        # return [self.card(c['color'], c['rank']) for hand in self.clues for c in hand]
        # sort, s.t. agents cards are at index 0
        card_knowledge.insert(0, card_knowledge.pop(card_knowledge.index(card_knowledge[self.players.index(self.agent_name)])))
        return card_knowledge

    def get_legal_moves(self):
        # order is 1. discard 2. play 3. reveal_color reveal rank and RYGWB for color
        legal_moves = []

        # discard if possible
        if self.information_tokens < self.max_info_tokens:
            for i in range(self.cards_per_hand):
                legal_moves.append({'action_type': 'DISCARD', 'card_index': i})
        # play
        for i in range(self.cards_per_hand):
            legal_moves.append({'action_type': 'PLAY', 'card_index': i})

        # clue if info token available
        if self.information_tokens > 0:
            hand_list = self.get_sorted_hand_list()

            # append colors
            for i in range(1, self.num_players):

                colors = set()
                for card in hand_list[i]:
                    print(card, type(card))
                    colors.add(card['color'])

                colors = self._sort_colors(colors)
                for c in colors:
                    legal_moves.append({'action_type': 'REVEAL_COLOR', 'target_offset': i, 'color': c})

            # append ranks
            for i in range(1, self.num_players):
                ranks = set()
                for card in hand_list[i]:
                    ranks.add(card['rank'])
                for r in ranks:
                    legal_moves.append({'action_type': 'REVEAL_Rank', 'target_offset': i, 'rank': r})

        return legal_moves

    @staticmethod
    def _sort_colors(colors: Set) -> List:
        """ Sorts list, s.t. colors are in order RYGWB """
        result = list()
        for i in range(len(colors)):
            if 'R' in colors:
                colors.remove('R')
                result.append('R')
            if 'Y' in colors:
                colors.remove('Y')
                result.append('Y')
            if 'G' in colors:
                colors.remove('G')
                result.append('G')
            if 'W' in colors:
                colors.remove('W')
                result.append('W')
            if 'B' in colors:
                colors.remove('B')
                result.append('B')

        return result

    @staticmethod
    def parse_rank(rank):
        """ Returns rank as expected by the server """
        if int(rank) > -1:
            rank += 1
        return str(rank)

    def parse_action_to_msg(self, action: dict) -> str:
        """ Returns action message that the server can read. """
        # one of ['REVEAL_COLOR', 'REVEAL_RANK', 'PLAY', 'DISCARD']
        action_type = action['action_type']
        print('PLAYER TO ACT IS ACCORDING TO GAME:')
        print('---------')
        print(self.players.index(self.agent_name))
        print('---------')
        print(action)
        # return value
        a = ''

        # -------- Convert CLUES ----------- #
        if action_type == 'REVEAL_COLOR':
            type = '0'  # 0 for type 'CLUE'
            target_offset = action['target_offset']
            # compute absolute player position from target_offset
            target = str(self.next_player(offset=target_offset))
            cluetype = '1'  # 1 for COLOR clue
            cluevalue = str(self.convert_color(action['color']))

            a = 'action {"type":'+type+',"target":'+target+',"clue":{"type":'+cluetype+',"value":'+cluevalue+'}}'

        if action_type == 'REVEAL_RANK':
            type = '0' # 0 for type 'CLUE'
            target_offset = action['target_offset']
            # compute absolute player position from target_offset
            target = self.next_player(offset=target_offset)
            cluetype = '0'  # 0 for RANK clue
            cluevalue = self.parse_rank(action['rank'])

            a = 'action {"type":' + type + ',"target":' + target + ',"clue":{"type":' + cluetype + ',"value":' + cluevalue + '}}'

        # -------- Convert PLAY ----------- #
        if action_type == 'PLAY': # 1 for type 'PLAY'
            type = '1'
            card_index = action['card_index']
            # target is referenced by absolute card number, gotta convert from given index
            target = str(self.card_numbers[self.players.index(self.agent_name)][card_index])

            a = 'action {"type":' + type + ',"target":' + target + '}'

        # -------- Convert DISCARD ----------- #
        if action_type == 'DISCARD':
            type = '2' # 2 for type 'DISCARD'
            card_index = action['card_index']
            # target is referenced by absolute card number, gotta convert from given index
            target = str(self.card_numbers[self.players.index(self.agent_name)][card_index])

            a = 'action {"type":' + type + ',"target":' + target + '}'
        print('action')
        print(a)
        return a

    def reset(self):
        self.hand_list = list()
        self.clues = list()
        self.card_numbers = list()
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        self.information_tokens = self.max_info_tokens
        self.life_tokens = self.max_life_tokens
        self.discard_pile = list()
        self.last_moves = list()
        return

    """ 
    # ------------------------------------------------- # ''
    # ------------------ MOCK METHODS ----------------  # ''
    # ------------------------------------------------- # ''    
    """

    def _get_pyhanabi_move_mock(self, dict_action):
        type = self._get_move_type(dict_action)
        card_index = None
        target_offset = None
        color = None
        rank = None
        discard_move = None
        play_move = None
        reveal_color_move = None
        reveal_rank_move = None

        move = HanabiMoveMock(
            type=type,
            card_index=card_index,
            target_offset=target_offset,
            color=color,
            rank=rank,
            discard_move=discard_move,
            play_move=play_move,
            reveal_color_move=reveal_color_move,
            reveal_rank_move=reveal_rank_move
        )
        return move

    @staticmethod
    def _get_move_type(move):
        """
            Return Move types, consistent with hanabi_lib/hanabi_move.h.
            INVALID = 0
            PLAY = 1
            DISCARD = 2
            REVEAL_COLOR = 3
            REVEAL_RANK = 4
            DEAL = 5
        """
        if move['type'] == 'play':
            return 1
        elif move['type'] == 'discard':
            return 2
        elif move['type'] == 'clue':
            if move['clue']['type'] == 0:  # rank clue
                return 4
            else:
                return 3
        elif move['type'] == 'draw':
            return 5

        return


""" 
    # ------------------------------------------------- # ''
    # ------------------ MOCK Classes ----------------  # ''
    # ------------------------------------------------- # ''    
"""

""" These are used to wrap the low level interface implemented in pyhanabi.py """


class HanabiHistoryItemMock:
    """ Just a mock, see mock method section for details """
    def __init__(self, move, player, scored, information_token, color, rank, card_info_revealed, card_info_newly_revealed, deal_to_player):
        self._move = move
        self._player = player
        self._scored = scored
        self._information_token = information_token
        self._color = color
        self._rank = rank
        self._card_info_revealed = card_info_revealed
        self._card_info_newly_revealed = card_info_newly_revealed
        self._deal_to_player = deal_to_player

    def move(self):
        return self._move

    def player(self):
        return self._player

    def scored(self):
        return self._scored

    def information_token(self):
        return self._information_token

    def color(self):
        return self._color

    def rank(self):
        return self._rank

    def card_info_revealed(self):
        return self._card_info_revealed

    def card_info_newly_revealed(self):
        return self._card_info_newly_revealed

    def deal_to_player(self):
        return self._deal_to_player


class HanabiMoveMock:
    """ Just a mock, see mock method section for details """
    def __init__(self, type, card_index, target_offset, color, rank, discard_move, play_move, reveal_color_move, reveal_rank_move):
        self._type = type
        self._card_index = card_index
        self._target_offset = target_offset
        self._color = color
        self._rank = rank
        self._discard_move = discard_move
        self._play_move = play_move
        self._reveal_color_move = reveal_color_move
        self._reveal_rank_move = reveal_rank_move

    def type(self):
        return self._type

    def card_index(self):
        return self._card_index

    def target_offset(self):
        return self._target_offset

    def color(self):
        return self._color

    def rank(self):
        return self._rank

    def get_discard_move(self):
        return self._discard_move

    def get_play_move(self):
        return self._play_move

    def get_reveal_color_move(self):
        return self._reveal_color_move

    def get_reveal_rank_move(self):
        return self._reveal_rank_move

    def to_dict(self):
        pass


