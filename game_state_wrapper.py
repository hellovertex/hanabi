import ast
import typing

class GameStateWrapper:

    def __init__(self):
        self.players = None  # list of the players currently ingame
        self.num_players = None  # number of players ingame
        self.deck_size = None  # number of remaining cards in the deck
        self.cur_player = None  # int position of current player
        self.hand_list = list()  # is refreshed on each notify message
        self.observed_hands = list()  # is computed from hand_list on demand
        self.card_knowledge = list()  # is refreshed on each notify message
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        self.life_tokens = 3
        # ------- OBS Dict ----------#
        self.observations = dict()

    def init_players(self, notify_msg: str):
        """ Sets self.players to a list of the players currently ingame and creates empty hands """
        player_dict = ast.literal_eval(notify_msg.split('init')[1].replace('false', 'False').replace('list', 'List'))
        self.players = player_dict['names']
        self.num_players = len(players)
        self.hand_list = [list() for _ in num_players]

    def deal_cards(self, notify_msg):
        """ Initializes self.hand_list from server message 'notifyList [{"type":"draw","who":0,"rank":4,"suit":1,
        "order":0},...'"""

        # list of dictionaries storing the draws
        card_list = ast.literal_eval(notify_msg.split('notifyList ')[1].replace('false', 'False').replace('list', 'List'))

        for d in card_list:
            if d['type'] == 'draw':  # add card to hand of player with id d['who'] from left to right
                self.hand_list[d['who']].insert(0, self.card(d['suite'], d['rank']))

            if d['type'] == 'turn':
                # notifyList message also contains info on who goes first
                self.cur_player = d['who']

    def update_state(self, notify_msg):
        """ Notify message can contain 'turn', 'draw', 'play', 'clue', 'discard' as "type"-values"""
        'Create dictionary that contains action-type values'
        d = ast.literal_eval(notify_msg.split('notify')[1]).replace('false', 'False').replace('list', 'List')

        # TURN - set current player
        if d['type'] == 'turn':
            self.cur_player = d['who']
            self.observations['current_player'] = d['who']

        # DISCARD - on discard, remove the card from the players hand
        if d['type'] == 'discard':
            pid = d['which']['index']
            self.hand_list[pid].remove(card(d['which']['color'], d['which']['rank']))

        # DRAW - if player with pid draws a card, it is prepended to hand_list[pid]
        if d['type'] == 'draw':
            pid = d['who']
            self.hand_list[pid].insert(0, card(d['suite'], d['rank']))

        # PLAY - remove played card from players hand and update fireworks/life tokens
        if d['type'] == 'play':
            # remove card
            pid = d['which']['index']
            self.hand_list[pid].remove(card(d['which']['index'], d['which']['rank']))

            # update fireworks and life tokens eventually
            self.play(card(d['which']['index'], d['which']['rank']))

    def play(self, card):
        # on success, update fireworks
        if self.fireworks[card['color']] == card['rank']:
            self.fireworks[card['color']] += 1
        # on fail, remove a life token
        else:
            self.life_tokens -= 1

    def get_observation(self, agent_id):
        return self.observations['player_observations'][agent_id]

    @staticmethod
    def card(suite: int, rank: int):
        if rank > -1:  # return rank = -1 for an own unclued card
            rank -= 1  # server cards are not 0-indexed
        return {'color': convert_suite(suite), 'rank': rank}

    @staticmethod
    def convert_suite(suite: int):
        """
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
        """
        if suite == 0: return 'B'
        if suite == 1: return 'G'
        if suite == 2: return 'Y'
        if suite == 3: return 'R'
        if suite == 4: return 'W'
