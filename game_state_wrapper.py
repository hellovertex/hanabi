import ast
from typing import Optional


class GameStateWrapper:

    def __init__(self, num_players):
        """
        # ################################################ #
        # -------------------- CONFIG -------------------- #
        # ################################################ #
        """
        self.players = None  # list of names of players currently ingame
        self.num_players = num_players  # number of players ingame
        self.deck_size = None  # number of remaining cards in the deck
        self.life_tokens = 3  # todo get from config
        self.information_tokens = 8  # todo get from config
        self.deck_size = 50  # todo get from config

        # absolute reference
        self.cur_player = None  # int position of current player

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
        # number of card drawn next, starting at zero
        self.num_next_drawn_card = 0

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

    def init_players(self, notify_msg: str):
        """ Sets self.players to a list of the players currently ingame and creates empty hands """
        self.reset()
        player_dict = ast.literal_eval(notify_msg.split('init ')[1].replace('false', 'False').replace('list', 'List'))
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
        card_list = ast.literal_eval(notify_msg.split('notifyList ')[1].replace('false', 'False').replace('list', 'List'))

        for d in card_list:
            if d['type'] == 'draw':  # add card to hand of player with id d['who'] from left to right
                self.hand_list[d['who']].insert(0, self.card(d['suit'], d['rank']))
                self.deck_size -= 1
                # unfortunately, server references clued cards by an absolute number and not its index on the hand
                # so we store the number too, to map it onto indices for playing and discarding
                self.card_numbers[d['who']].insert(0, self.num_next_drawn_card)
                self.num_next_drawn_card += 1
                self.clues[d['who']].append({'color': None, 'rank': None})
            if d['type'] == 'turn':
                # notifyList message also contains info on who goes first
                self.cur_player = d['who']
        assert self.cur_player is not None

        return

    def update_state(self, notify_msg):
        """
        Updates Game State after server sends an action-notification.
        Notify message can contain 'turn', 'draw', 'play', 'clue', 'discard' as "type"-values
        """

        'Create dictionary from server message that contains actions'
        d = ast.literal_eval(notify_msg.split('notify ')[1].replace('false', 'False').replace('list', 'List'))
        print(d['type'])
        print(self.hand_list)
        print(self.card_numbers)
        print(self.clues)
        # TURN - set current player
        if d['type'] == "turn":
            self.cur_player = d['who']
        # DISCARD - on discard, remove the card from the players hand
        if d['type'] == 'discard':
            pid = d['which']['index']
            c = self.card(d['which']['color'], d['which']['rank'])
            c_idx = self.hand_list.index(c)
            self.hand_list[pid].remove(c)
            del self.card_numbers[pid][c_idx]
            self.discard_pile.append(c)
            # update self.clues
            del self.clues[pid][c_idx]

        # DRAW - if player with pid draws a card, it is prepended to hand_list[pid]
        if d['type'] == 'draw':
            pid = d['who']
            self.hand_list[pid].insert(0, self.card(d['suit'], d['rank']))
            self.deck_size -= 1
            # unfortunately, server references clued cards by an absolute number and not its index on the hand
            # so we store the number too, to map it onto indices for playing and discarding
            self.card_numbers[d['who']].insert(0, self.num_next_drawn_card)
            self.num_next_drawn_card += 1
            # update self.clues
            self.clues[d['who']].insert(0, {'color': None, 'rank': None})

        # PLAY - remove played card from players hand and update fireworks/life tokens
        if d['type'] == 'play':
            print("CARDNUMBERS")
            print(self.card_numbers)
            # remove card
            pid = d['which']['index']
            c = self.card(d['which']['index'], d['which']['rank'])
            c_idx = self.hand_list.index(c)
            self.hand_list[pid].remove()
            del self.card_numbers[pid][c_idx]
            # update self.clues
            del self.clues[pid][c_idx]
            # update fireworks and life tokens eventually
            self.play(c)

        # CLUE - change players card_knowledge and remove an info-token
        if d['type'] == 'clue':
            self.update_clues(d)
            self.information_tokens -= 1  # validity has previously been checked by the server so were good with that

        # Add to history
        if d['type'] in ['play', 'discard', 'clue']:
            self.cur_player = self.next_player(offset=1)
        # self.append_to_last_moves(d)

        return

    def update_clues(self, dict_clue):
        clue = dict_clue['clue']
        target = dict_clue['target']
        touched_cards = dict_clue['List']
        for c in touched_cards:
            idx_c = self.card_numbers[target].index(c)
            if clue['type'] == 0:
                self.clues[target][idx_c]['rank'] = clue['value']
            else:
                self.clues[target][idx_c]['color'] = clue['value']
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
        pass

    def get_observation(self, agent_id):
        """ Will always return observation of the calling agent as this is the only thing needed. We just use agent_id
        for consistency and readability"""
        assert agent_id == self.cur_player
        observation = {
            'current_player': agent_id,
            'current_player_offset': 0,
            'life_tokens': self.life_tokens,
            'information_tokens': self.information_tokens,
            'num_players': self.num_players,
            'deck_size': self.deck_size,
            'fireworks': self.fireworks,
            'legal_moves': None,  # not gon compute these here, as our agents compute their moves anyway
            'observed_hands': self.hand_list,
            'discard_pile': self.discard_pile,
            'card_knowledge': self.get_card_knowledge(),
            'vectorized': None,  # Currently not needed, we can implement it later on demand
            'last_moves': self.last_moves  # actually not contained in the returned dict of the
            # rl_env.HanabiEnvobservation._extract_from_dict method, but we need a history so we add this here.
            # Similarly, it can be added by appending obs_dict['last_moves'] = observation.last_moves() in said method.
        }
        return observation

    def card(self, suit: int, rank: int):
        """ Returns card format desired by agent"""
        if rank is not None:
            if rank > -1:  # return rank = -1 for an own unclued card
                rank -= 1  # server cards are not 0-indexed
        return {'color': self.convert_suit(suit), 'rank': rank}

    def next_player(self, offset):
        return divmod(self.cur_player + int(offset), self.num_players)[1]

    @staticmethod
    def convert_suit(suit: int) -> Optional[str]:

        """
        Returns format desired by agent
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
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

    def get_card_knowledge(self):
        """ Returns self.clues but formatted in a way desired by the agent"""
        card_knowledge = list()

        for hand in self.clues:
            h = list()
            for c in hand:
                h.append(self.card(c['color'], c['rank']))
            card_knowledge.append(h)
        # return [self.card(c['color'], c['rank']) for hand in self.clues for c in hand]
        return card_knowledge

    @staticmethod
    def parse_rank(rank):
        if int(rank) > -1:
            rank += 1
        return str(rank)

    def parse_action_to_msg(self, action: dict) -> str:
        """ Returns action message that the server can read. """
        # one of ['REVEAL_COLOR', 'REVEAL_RANK', 'PLAY', 'DISCARD']
        action_type = action['action_type']

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
            a = 'action {"type":'+type+',"target":'+target+',"clue":{"type":'+cluetype+',"value:":'+cluevalue+'}}'

        if action_type == 'REVEAL_RANK':
            type = '0' # 0 for type 'CLUE'
            target_offset = action['target_offset']
            # compute absolute player position from target_offset
            target = self.next_player(offset=target_offset)
            cluetype = '0'  # 0 for RANK clue
            cluevalue = self.parse_rank(action['rank'])
            a = 'action {"type":' + type + ',"target":' + target + ',"clue":{"type":' + cluetype + ',"value:":' + cluevalue + '}}'

        # -------- Convert PLAY ----------- #
        if action_type == 'PLAY': # 1 for type 'PLAY'
            type = '1'
            card_index = action['card_index']
            # target is referenced by absolute card number, gotta convert from given index
            target = str(self.card_numbers[self.cur_player][card_index])
            a = 'action {"type":' + type + ',"target":' + target + '}'

        # -------- Convert DISCARD ----------- #
        if action_type == 'DISCARD':
            type = '2' # 2 for type 'DISCARD'
            card_index = action['card_index']
            # target is referenced by absolute card number, gotta convert from given index
            target = str(self.card_numbers[self.cur_player][card_index])
            a = 'action {"type":' + type + ',"target":' + target + '}'

        return a

    def reset(self):
        self.hand_list = list()
        self.clues = list()
        self.card_numbers = list()
        self.num_next_drawn_card = -1
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        self.discard_pile = list()
        self.last_moves = list()
        return