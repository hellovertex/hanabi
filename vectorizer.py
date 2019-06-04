import numpy as np

np.set_printoptions(threshold=np.inf)
import pyhanabi as utils

COLOR_CHAR = ["R", "Y", "G", "W", "B"]


def color_plausible(color,
                    player_card_knowledge,
                    card_id,
                    card_color_revealed,
                    last_hand,
                    last_player_action,
                    last_player_card_knowledge):
    plausible = True
    if last_hand:
        if last_player_action == "PLAY" or last_player_action == "DISCARD":
            player_card_knowledge = player_card_knowledge + last_player_card_knowledge
            if card_id == (len(last_player_card_knowledge) - 1):
                return plausible

    color = utils.color_idx_to_char(color)
    if card_color_revealed:
        if color == player_card_knowledge[card_id]["color"]:
            return plausible
        else:
            plausible = False
            return plausible

    for i, card in enumerate(player_card_knowledge):

        tmp_color = card["color"]
        if tmp_color == color:
            plausible = False

    return plausible


def rank_plausible(rank,
                   player_card_knowledge,
                   card_id,
                   card_rank_revealed,
                   last_hand,
                   last_player_action,
                   last_player_card_knowledge):
    plausible = True
    if last_hand:
        if last_player_action == "PLAY" or last_player_action == "DISCARD":
            player_card_knowledge = player_card_knowledge + last_player_card_knowledge
            if card_id == (len(player_card_knowledge) - 1):
                return plausible

    if card_rank_revealed:
        if rank == player_card_knowledge[card_id][rank]:
            return True
        else:
            return False

    for i, card in enumerate(player_card_knowledge):

        if card["rank"] == rank:
            plausible = False

    return plausible


'''
Used to vectorize/encode player-dependent state-dicts and action dicts that are used
for agents that where trained with Hanabi Game Environment
For more details check:
GitHub Wiki - Hanabi Env Doc - Encodings
'''


class ObservationVectorizer(object):

    def __init__(self, env):
        '''
        Encoding Order =
         HandEncoding
        +BoardEncoding
        +DiscardEncoding
        +LastAcionEncoding
        +CardKnowledgeEncoding
        '''
        self.env = env
        self.obs = None
        self.num_players = self.env.num_players()
        self.num_colors = self.env.num_colors()
        self.num_ranks = self.env.num_ranks()
        self.hand_size = self.env.hand_size()
        self.max_info_tokens = self.env.max_information_tokens()
        self.max_life_tokens = self.env.max_life_tokens()
        self.max_moves = self.env.max_moves()
        self.bits_per_card = self.num_colors * self.num_ranks
        self.max_deck_size = 0
        # start of the vectorized observation
        self.offset = None

        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                self.max_deck_size += self.env.num_cards(color, rank)
        """ Bit lengths """
        # Compute total state length
        self.hands_bit_length = (self.num_players - 1) * self.hand_size * self.bits_per_card + self.num_players
        # print("hand encoding section in range: {}-{}".format(0,self.hands_bit_length))

        self.board_bit_length = self.max_deck_size - self.num_players * \
                                self.hand_size + self.num_colors * self.num_ranks \
                                + self.max_info_tokens + self.max_life_tokens
        # print("board encoding section in range: {}-{}".format(self.hands_bit_length,self.hands_bit_length+self.board_bit_length))

        self.discard_pile_bit_length = self.max_deck_size
        # print("discard pile encoding in range: {}-{}".format(self.hands_bit_length+self.board_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length))

        self.last_action_bit_length = self.num_players + 4 + self.num_players + \
                                      self.num_colors + self.num_ranks \
                                      + self.hand_size + self.hand_size + self.bits_per_card + 2
        # print("last action encoding in range {}-{}".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length))
        self.card_knowledge_bit_length = self.num_players * self.hand_size * \
                                         (self.bits_per_card + self.num_colors + self.num_ranks)
        # print("card knowledge encoding in range: {}-{}\n".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length+self.card_knowledge_bit_length))
        self.total_state_length = self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length \
                                  + self.last_action_bit_length + self.card_knowledge_bit_length
        self.obs_vec = np.zeros(self.total_state_length)


        self.last_player_action = None

    def get_vector_length(self):
        return self.total_state_length

    def vectorize_observation(self, obs):
        self.obs = obs
        if obs["last_moves"][0] != "DEAL":
            self.last_player_action = obs["last_moves"][0]
        else:
            self.last_player_action = obs["last_moves"][1]
        self.encode_hands(obs)
        self.encode_board(obs)
        self.encode_discards(obs)
        self.encode_last_action()
        self.encode_card_knowledge(obs)

        return self.obs_vec

    def encode_hands(self, obs):
        self.offset = 0
        # don't use own hand
        hands = obs["observed_hands"]
        for player_hand in hands:
            if player_hand[0]["color"] is not None:
                num_cards = 0
                for card in player_hand:
                    rank = card["rank"]
                    color = utils.color_char_to_idx(card["color"])
                    card_index = color * self.num_ranks + rank
                    self.obs_vec[self.offset + card_index] = 1
                    num_cards += 1
                    self.offset += self.bits_per_card
                if num_cards < self.hand_size:
                    self.offset += (self.hand_size - num_cards) * self.bits_per_card

        # For each player, set a bit if their hand is missing a card
        for i, player_hand in enumerate(hands):
            if len(player_hand) < self.hand_size:
                self.obs_vec[self.offset + i] = 1
            self.offset += 1

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

    def encode_discards(self, obs):
        discard_pile = obs["discard_pile"]
        counts = np.zeros(self.num_colors * self.num_ranks)
        for card in discard_pile:
            color = utils.color_char_to_idx(card["color"])
            rank = card["rank"]
            counts[color * self.num_ranks + rank] += 1

        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                num_discarded = counts[c * self.num_ranks + r]
                for i in range(int(num_discarded)):
                    self.obs_vec[self.offset + i] = 1
                self.offset += self.env.num_cards(c, r)

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length) == 0
        return True

    def encode_last_action(self):
        if self.last_player_action is None:
            self.offset += self.last_action_bit_length
        else:
            last_move_type = self.last_player_action["type"]
            # print("Last Move Type: {}".format(last_move_type))
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
                observer_relative_target = (self.last_player_action["player"] + self.last_player_action[
                    "target_offset"]) % self.num_players
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
                card_index = self.last_player_action["hand_card_id"]
                self.obs_vec[self.offset + card_index] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index_hgame = utils.color_char_to_idx(self.last_player_action["color"]) * self.num_ranks + \
                                   self.last_player_action["rank"]
                print(self.offset + card_index_hgame)
                self.obs_vec[self.offset + card_index_hgame] = 1

            self.offset += self.bits_per_card

            if last_move_type == "PLAY":
                if self.last_player_action["scored"]:
                    self.obs_vec[self.offset] = 1

                # IF INFO TOKEN WAS ADDED
                if self.last_player_action["information_token"]:
                    self.obs_vec[self.offset + 1] = 1

            self.offset += 2

        assert self.offset - (
                self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length) == 0
        return True

    def encode_card_knowledge(self, obs):

        card_knowledge_list = obs["card_knowledge"]

        for ih, player_card_knowledge in enumerate(card_knowledge_list):
            num_cards = 0

            # print("##########################################")
            # print("############### PLAYER {} ################".format(ih+1))
            # print("##########################################\n")

            ### TREAT LAST HAND EDGE CASES
            if ih == self.num_players - 1:
                last_hand = True
            else:
                last_hand = False

            for card_id, card in enumerate(player_card_knowledge):

                # print("##########################################")
                # print("tmp-card id: {}".format(card_id))
                # print("##########################################")

                if card["color"] is not None:
                    card_color_revealed = True
                else:
                    card_color_revealed = False
                if card["rank"] is not None:
                    card_rank_revealed = True
                else:
                    card_rank_revealed = False

                last_player_card_knowledge = card_knowledge_list[1]

                for color in range(self.num_colors):

                    if color_plausible(color,
                                       player_card_knowledge,
                                       card_id,
                                       card_color_revealed, last_hand,
                                       self.last_player_action["type"], last_player_card_knowledge):

                        for rank in range(self.num_ranks):

                            if rank_plausible(rank,
                                              player_card_knowledge,
                                              card_id,
                                              card_rank_revealed,
                                              last_hand,
                                              self.last_player_action["type"],
                                              last_player_card_knowledge):
                                card_index = color * self.num_ranks + rank

                                # if ((self.offset+card_index) in error_list):
                                #     print(self.offset+card_index)
                                #     print("\nFailed encoded card: {}, with index: {}, at hand_index: {}".format(card, card_id, ih))
                                #     print("Wrongly assigned 'plausible' to color: {}, rank: {}\n".format(utils.color_idx_to_char(color),rank))
                                self.obs_vec[self.offset + card_index] = 1

                self.offset += self.bits_per_card

                # Encode explicitly revealed colors and ranks
                if card["color"] is not None:
                    color = utils.color_char_to_idx(card["color"])

                    self.obs_vec[self.offset + color] = 1

                self.offset += self.num_colors

                if card["rank"] is not None:
                    rank = card["rank"]
                    self.obs_vec[self.offset + rank] = 1

                self.offset += self.num_ranks
                num_cards += 1

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)


        assert self.offset - (
                    self.hands_bit_length +
                    self.board_bit_length +
                    self.discard_pile_bit_length +
                    self.last_action_bit_length +
                    self.card_knowledge_bit_length) == 0

        return True


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
        self.num_players = self.env.num_players()
        self.num_ranks = self.env.num_ranks()
        self.num_colors = self.env.num_colors()
        self.hand_size = self.env.hand_size()
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