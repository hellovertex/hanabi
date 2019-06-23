import numpy as np
np.set_printoptions(threshold=np.inf)
import pyhanabi as utils
import copy

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

class HandKnowledge(object):
    def __init__(self, hand_size, num_ranks, num_colors):
        self.num_ranks = num_ranks
        self.num_colors = num_colors
        self.hand = [CardKnowledge(num_ranks, num_colors) for _ in range(hand_size)]

    def sync_colors(self, card_ids, color):

        for cid in range(len(self.hand)):
            if cid in card_ids:
                for c in range(len(self.hand[cid].colors)):
                    if c != color:
                        self.hand[cid].colors[c] = None
            else:
                self.hand[cid].colors[color] = None

    def sync_ranks(self, card_ids, rank):

        for cid in range(len(self.hand)):
            if cid in card_ids:
                for r in range(len(self.hand[cid].ranks)):
                    if r != rank:
                        self.hand[cid].ranks[r] = None
            else:
                self.hand[cid].ranks[rank] = None

    def remove_card(self, card_id, deck_empty = False):
        new_hand = []
        for c_id,card in enumerate(self.hand):
            if c_id != card_id:
                new_hand.append(card)

        # NOTE: STOP REFILLING IF DECK IS EMPTY
        if not deck_empty:
            while len(new_hand) < len(self.hand):
                new_hand.append(CardKnowledge(self.num_ranks, self.num_colors))

        self.hand = new_hand

class CardKnowledge(object):
    def __init__(self, num_ranks, num_colors):
        self.colors = [c for c in range(num_colors)]
        self.ranks = [r for r in range(num_ranks)]

    def color_plausible(self, color):
        # print("\n INSIDE COLOR PLAUSIBLE FUNCTION")
        # print(self.colors[color])
        # print("\n")
        return self.colors[color] != None

    def rank_plausible(self, rank):
        # print("\n INSIDE RANK PLAUSIBLE FUNCTION")
        # print(self.ranks[rank])
        # print("\n")
        return self.ranks[rank] != None

    def remove_rank(self, rank):
        if rank in self.ranks:
            self.ranks[rank] = None

    def remove_color(self, color):
        if color in self.colors:
            self.colors[color] = None

'''
Used to vectorize/encode player-dependent state-dicts and action dicts that are used
for agents that where trained with Hanabi Game Environment
For more details check:
GitHub Wiki - Hanabi Env Doc - Encodings
'''

class LegalMovesVectorizer(object):
    '''
    // Uid mapping.  h=hand_size, p=num_players, c=colors, r=ranks
    // 0, h-1: discard
    // h, 2h-1: play
    // 2h, 2h+(p-1)c-1: color hint
    // 2h+(p-1)c, 2h+(p-1)c+(p-1)r-1: rank hint
    '''
    def __init__(self, env):
        self.env = env
        self.num_players = self.env.game.num_players()
        self.num_ranks = self.env.game.num_ranks()
        self.num_colors = self.env.game.num_colors()
        self.hand_size = self.env.game.hand_size()
        self.max_reveal_color_moves = (self.num_players - 1) * self.num_colors
        self.num_moves = self.env.num_moves()

    def get_legal_moves_as_int(self, legal_moves):
        legal_moves_as_int = [-np.Inf for _ in range(self.num_moves)]
        tmp_legal_moves_as_int = [self.get_move_uid(move) for move in legal_moves]

        for move in tmp_legal_moves_as_int:
            legal_moves_as_int[move] = 0.0

        return [self.get_move_uid(move) for move in legal_moves]

    def get_legal_moves_as_int_formated(self,legal_moves_as_int):

        new_legal_moves = np.full(self.num_moves, -float('inf'))

        if legal_moves_as_int:
            new_legal_moves[legal_moves_as_int] = 0
        return new_legal_moves

    def get_move_uid(self, move):
        if move["action_type"] == "DISCARD":
            card_index = move["card_index"]
            return card_index

        elif move["action_type"] == "PLAY":
            card_index = move["card_index"]
            return self.hand_size + card_index

        elif move["action_type"] == "REVEAL_COLOR":
            target_offset = move["target_offset"]
            color = utils.color_char_to_idx(move["color"])
            return self.hand_size + self.hand_size + (target_offset-1) * self.num_colors + color

        elif move["action_type"] == "REVEAL_RANK":
            rank = move["rank"]
            target_offset = move["target_offset"]
            return self.hand_size + self.hand_size + self.max_reveal_color_moves + (target_offset-1) * self.num_ranks + rank
        else:
            return -1


class ObservationVectorizer(object):
    def __init__(self,env):
            '''
            Encoding Order =
             HandEncoding
            +BoardEncoding
            +DiscardEncoding
            +LastAcionEncoding
            +CardKnowledgeEncoding
            '''
            self.env = env

            self.num_players = self.env.game.num_players()
            self.num_colors = self.env.game.num_colors()
            self.num_ranks = self.env.game.num_ranks()
            self.hand_size = self.env.game.hand_size()
            self.max_info_tokens = self.env.game.max_information_tokens()
            self.max_life_tokens = self.env.game.max_life_tokens()
            self.max_moves = self.env.game.max_moves()
            self.bits_per_card = self.num_colors * self.num_ranks
            self.max_deck_size = 0
            for color in range(self.num_colors):
                for rank in range(self.num_ranks):
                    self.max_deck_size += self.env.game.num_cards(color,rank)

            # Compute total state length
            self.hands_bit_length = (self.num_players - 1) * self.hand_size * \
                            self.bits_per_card + self.num_players
            print("hand encoding section in range: {}-{}".format(0,self.hands_bit_length))

            self.board_bit_length = self.max_deck_size - self.num_players * \
                            self.hand_size + self.num_colors * self.num_ranks \
                                + self.max_info_tokens + self.max_life_tokens
            print("board encoding section in range: {}-{}".format(self.hands_bit_length,self.hands_bit_length+self.board_bit_length))

            self.discard_pile_bit_length = self.max_deck_size
            print("discard pile encoding in range: {}-{}".format(self.hands_bit_length+self.board_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length))

            self.last_action_bit_length = self.num_players + 4 + self.num_players + \
                            self.num_colors + self.num_ranks \
                                + self.hand_size + self.hand_size + self.bits_per_card + 2
            print("last action encoding in range {}-{}".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length))
            self.card_knowledge_bit_length = self.num_players * self.hand_size *\
                            (self.bits_per_card + self.num_colors + self.num_ranks)
            print("card knowledge encoding in range: {}-{}\n".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length+self.card_knowledge_bit_length))
            self.total_state_length = self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length \
                                    + self.last_action_bit_length + self.card_knowledge_bit_length
            self.obs_vec = np.zeros(self.total_state_length)

            self.last_player_card_knowledge = []

            self.player_knowledge = [HandKnowledge(self.hand_size, self.num_ranks, self.num_colors) for _ in range(self.num_players)]

            self.player_id = 0

            self.last_player_action = []

    def get_vector_length(self):
        return self.total_state_length

    # NOTE FOR NOW JUST PARSING VERY LAST 2 MOVES
    def parse_last_moves(self,last_moves):

        last_moves_dict = last_moves.move().to_dict()

        if last_moves_dict["action_type"] == "REVEAL_COLOR" or last_moves_dict["action_type"] == "REVEAL_RANK":
            color = utils.color_idx_to_char(last_moves.move().color())
            rank = last_moves.move().rank()

            last_moves_dict.update([
                ("color",color),
                ("rank",rank)
            ])

        elif last_moves_dict["action_type"] == "PLAY" or last_moves_dict["action_type"] == "DISCARD":
            color = utils.color_idx_to_char(last_moves.color())
            rank = last_moves.rank()

            last_moves_dict.update([
                ("color",color),
                ("rank",rank)
            ])

        last_moves_dict.update([
            ("player", last_moves.player()),
            ("scored", last_moves.scored()),
            ("information_token", last_moves.information_token()),
            ("card_info_revealed", last_moves.card_info_revealed())
        ])

        last_moves_dict = [last_moves_dict]
        return last_moves_dict

    # NOTE: change back to what it was before
    def vectorize_observation(self,obs):
        # if len(obs["last_moves"]) > 2:
            # print(obs["last_moves"][1])
        self.obs_vec = np.zeros(self.total_state_length)
        self.obs = obs
        if obs["last_moves"] != []:

            o = copy.copy(obs)

            obs["last_moves"] = self.parse_last_moves(obs["last_moves"][0])

            if obs["last_moves"][0]["action_type"] != "DEAL":
                self.last_player_action = obs["last_moves"][0]

            elif len(o["last_moves"]) >= 2:
                obs["last_moves"] = self.parse_last_moves(o["last_moves"][1])
                self.last_player_action = obs["last_moves"][0]
            else:
                self.last_player_action = []

        self.encode_hands(obs)
        self.encode_board(obs)
        self.encode_discards(obs)
        self.encode_last_action(obs)
        self.encode_card_knowledge(obs)

        return self.obs_vec

    '''Enocdes cards in all other player's hands (excluding our unknown hand),
     and whether the hand is missing a card for all players (when deck is empty.)
     Each card in a hand is encoded with a one-hot representation using
     <num_colors> * <num_ranks> bits (25 bits in a standard game) per card.
     Returns the number of entries written to the encoding.'''

    def encode_hands(self,obs):

        self.offset = 0

        hands = obs["observed_hands"]
        for i,player_hand in enumerate(hands):
            if (player_hand[0]["color"] != None):

                num_cards = 0
                for card in player_hand:
                    # Only a player's own cards can be invalid/unobserved.
                    rank = card["rank"]
                    color = utils.color_char_to_idx(card["color"])
                    card_index = color * self.num_ranks + rank

                    self.obs_vec[self.offset + card_index] = 1
                    num_cards += 1
                    self.offset += self.bits_per_card

                '''
                A player's hand can have fewer cards than the initial hand size.
                Leave the bits for the absent cards empty (adjust the offset to skip
                bits for the missing cards).
                '''
                if num_cards < self.hand_size:

                    self.offset += (self.hand_size - num_cards) * self.bits_per_card

        # For each player, set a bit if their hand is missing a card.
        i = 0
        for i,player_hand in enumerate(hands):
            if len(player_hand) < self.hand_size:

                self.obs_vec[self.offset+i] = 1

        self.offset += self.num_players

        assert self.offset - self.hands_bit_length == 0
        return True

    def encode_board(self, obs):
        # encode the deck size:
        for i in range(obs["deck_size"]):
            self.obs_vec[self.offset + i] = 1
        self.offset += self.max_deck_size - self.hand_size * self.num_players

        # encode fireworks
        fireworks = obs["fireworks"]
        for c in range(len(fireworks)):
            color = utils.color_idx_to_char(c)
            # print(fireworks[color])
            if fireworks[color] > 0:
                self.obs_vec[self.offset + fireworks[color] - 1] = 1
            self.offset += self.num_ranks

        # encode info tokens
        info_tokens = obs["information_tokens"]
        for t in range(info_tokens):
            self.obs_vec[self.offset + t] = 1
        self.offset += self.max_info_tokens

        # encode life tokens
        life_tokens = obs["life_tokens"]
        for l in range(life_tokens):
            self.obs_vec[self.offset + l] = 1
        self.offset += self.max_life_tokens

        assert self.offset - (self.hands_bit_length + self.board_bit_length) == 0
        return True

    def encode_discards(self,obs):
        discard_pile = obs["discard_pile"]
        counts = np.zeros(self.num_colors*self.num_ranks)
        for card in discard_pile:
            color = utils.color_char_to_idx(card["color"])
            rank = card["rank"]
            counts[color*self.num_ranks + rank] += 1

        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                num_discarded = counts[c*self.num_ranks+r]
                for i in range(int(num_discarded)):
                    self.obs_vec[self.offset+i] = 1
                self.offset += self.env.game.num_cards(c,r)

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length) == 0
        return True


    # NOTE: CHANGE ACTION TYPE BACK
    def encode_last_action(self,obs):
        # print("==============")
        # print(self.last_player_action)
        # print("==============")
        if self.last_player_action == []:
            self.offset += self.last_action_bit_length
        else:
            last_move_type = self.last_player_action["action_type"]
            self.obs_vec[self.offset + self.last_player_action["player"]] = 1
            self.offset += self.num_players

            if last_move_type == "PLAY":
                self.obs_vec[self.offset] = 1
            elif last_move_type == "DISCARD":
                self.obs_vec[self.offset + 1] = 1
            elif last_move_type == "REVEAL_COLOR":
                self.obs_vec[self.offset + 2] = 1
            elif last_move_type == "REVEAL_RANK":
                self.obs_vec[self.offset + 3] = 1
            else:
                print("ACTION UNKNOWN")
                return
            self.offset += 4

            # NEED TO COMPUTE RELATIVE OFFSET FROM CURRENT PLAYER
            if last_move_type == "REVEAL_COLOR" or last_move_type == "REVEAL_RANK":
                observer_relative_target = (self.last_player_action["player"] + self.last_player_action["target_offset"]) % self.num_players
                self.obs_vec[self.offset + observer_relative_target] = 1

            self.offset += self.num_players

            if last_move_type == "REVEAL_COLOR":
                last_move_color = self.last_player_action["color"]
                self.obs_vec[self.offset + utils.color_char_to_idx(last_move_color)] = 1

            self.offset += self.num_colors

            if last_move_type == "REVEAL_RANK":
                last_move_rank = self.last_player_action["rank"]
                self.obs_vec[self.offset + last_move_rank] = 1

            self.offset += self.num_ranks

            # If multiple positions where selected
            if last_move_type == "REVEAL_COLOR" or last_move_type == "REVEAL_RANK":
                positions = self.last_player_action["card_info_revealed"]
                for pos in positions:
                    self.obs_vec[self.offset + pos] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index = self.last_player_action["card_index"]
                self.obs_vec[self.offset + card_index] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index_hgame = utils.color_char_to_idx(self.last_player_action["color"]) * self.num_ranks + self.last_player_action["rank"]
                print(self.offset + card_index_hgame)
                self.obs_vec[self.offset + card_index_hgame] = 1

            self.offset += self.bits_per_card

            if last_move_type == "PLAY":
                if self.last_player_action["scored"]:
                    self.obs_vec[self.offset] = 1

                ### IF INFO TOKEN WAS ADDED
                if self.last_player_action["information_token"]:
                    self.obs_vec[self.offset + 1] = 1

            self.offset += 2

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length) == 0
        return True

    # TODO: NEED TO SYNC WITH player_card_knowledge

    def encode_card_knowledge(self,obs):

        hands = obs["observed_hands"]
        card_knowledge_list = obs["card_knowledge"]
        discard_pile_knowledge = obs["discard_pile"]
        fireworks_knowledge = obs["fireworks"]
        current_player_id = obs["current_player"]

        ### SYNC CARD KNOWLEDGE AFTER HINT GIVEN ###
        if self.last_player_action != []:
            if self.last_player_action["action_type"] == "REVEAL_COLOR":
                player_hand_to_sync = (self.last_player_action["player"] + self.last_player_action["target_offset"] + current_player_id) % self.num_players
                card_pos_to_sync = self.last_player_action["card_info_revealed"]
                color_to_sync = utils.color_char_to_idx(self.last_player_action['color'])
                # print("\n==============================")
                # print(f"SYNCING CARD KNOWLEDGE OF PLAYER: {player_hand_to_sync}")
                # print("================================\n")
                self.player_knowledge[player_hand_to_sync].sync_colors(card_pos_to_sync, color_to_sync)


        if self.last_player_action != []:
            if self.last_player_action["action_type"] == "REVEAL_RANK":
                player_hand_to_sync = (self.last_player_action["player"] + self.last_player_action["target_offset"] + current_player_id) % self.num_players
                card_pos_to_sync = self.last_player_action["card_info_revealed"]
                rank_to_sync = self.last_player_action['rank']
                # print("\n==============================")
                # print(f"SYNCING CARD KNOWLEDGE OF PLAYER: {player_hand_to_sync}")
                # print("================================\n")
                self.player_knowledge[player_hand_to_sync].sync_ranks(card_pos_to_sync, rank_to_sync)

        if self.last_player_action != []:
            if self.last_player_action["action_type"] == "PLAY" or self.last_player_action["action_type"] == "DISCARD":
                player_id = (self.last_player_action["player"] + current_player_id) % self.num_players
                card_id = self.last_player_action["card_index"]

                self.player_knowledge[player_id].remove_card(card_id)



        # print("========================")
        # print(f"LAST ACTION: {self.last_player_action}")
        # print("========================")

        # print("##########################################")
        # print(f"############### CURRENT PLAYER {current_player_id} ################")
        # print("##########################################\n")

        for ih, player_card_knowledge in enumerate(card_knowledge_list):
            num_cards = 0

            # print("##########################################")
            # print("############### TMP PLAYER CARD KNOWLEDGE ID {} ################".format(ih))
            # print(player_card_knowledge)
            # print("##########################################\n")

            rel_player_pos = (current_player_id + ih) % self.num_players

            for card_id, card in enumerate(player_card_knowledge):

                # print("\n###########################")
                # print("CARD")
                # print(card)
                # print("###########################\n")

                for color in range(self.num_colors):

                    if self.player_knowledge[rel_player_pos].hand[card_id].color_plausible(color):

                        for rank in range(self.num_ranks):

                            if self.player_knowledge[rel_player_pos].hand[card_id].rank_plausible(rank):

                                card_index = card_index = color * self.num_ranks + rank
                                # print("\n===============================")
                                # print(f"CURRENT PLAYER REAL: {current_player_id}, CURRENT PLAYER HAND: {ih}, CURRENT PLAYER RELATIVE {rel_player_pos}")
                                # print(f"OFFSET TO BE SET IN 555: {self.offset + card_index}")
                                # print(f"ASSIGNING PLAUSIBLE=TRUE TO CARD_ID: {card_id}, FOR COLOR: {color} AND RANK: {rank}")
                                # print("SET CARD KNOWLEDGE FOR THIS HAND")
                                # print("COLORS")
                                # print(self.player_knowledge[rel_player_pos].hand[0].colors,self.player_knowledge[rel_player_pos].hand[1].colors,self.player_knowledge[rel_player_pos].hand[2].colors,self.player_knowledge[rel_player_pos].hand[3].colors)
                                # print("RANKS")
                                # print(self.player_knowledge[rel_player_pos].hand[0].ranks,self.player_knowledge[rel_player_pos].hand[1].ranks,self.player_knowledge[rel_player_pos].hand[2].ranks,self.player_knowledge[rel_player_pos].hand[3].ranks)
                                # print("===============================\n")

                                self.obs_vec[self.offset + card_index] = 1


                self.offset += self.bits_per_card

                # Encode explicitly revealed colors and ranks
                if card["color"] != None:
                    color = utils.color_char_to_idx(card["color"])
                    # print("OFFSET TO BE SET IN 542")
                    # print(self.offset + color)
                    self.obs_vec[self.offset + color] = 1

                self.offset += self.num_colors

                if card["rank"] != None:
                    rank = card["rank"]

                    # print("OFFSET TO BE SET IN 551")
                    # print(self.offset + rank)

                    self.obs_vec[self.offset + rank] = 1

                self.offset += self.num_ranks
                num_cards += 1

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)

        self.last_player_card_knowledge = card_knowledge_list[0]

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length + self.card_knowledge_bit_length) == 0
        return True
