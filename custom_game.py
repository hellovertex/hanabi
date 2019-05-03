from __future__ import print_function

import numpy as np
import pyhanabi
from agents.custom_agent2 import CustomAgent

def run_game(game_parameters, human_players):
    def print_observation(observation):
        """Print some basic information about an agent observation."""
        print("--- Observation ---")
        print(observation)

        print("### Information about the observation retrieved separately ###")
        print("### Current player, relative to self: {}".format(
            observation.cur_player_offset()))
        print("### Observed hands: {}".format(observation.observed_hands()))
        print("### Card knowledge: {}".format(observation.card_knowledge()))
        print("### Discard pile: {}".format(observation.discard_pile()))
        print("### Fireworks: {}".format(observation.fireworks()))
        print("### Deck size: {}".format(observation.deck_size()))
        move_string = "### Last moves:"
        for move_tuple in observation.last_moves():
            move_string += " {}".format(move_tuple)
        print(move_string)
        print("### Information tokens: {}".format(observation.information_tokens()))
        print("### Life tokens: {}".format(observation.life_tokens()))
        print("### Legal moves: {}".format(observation.legal_moves()))
        print("--- EndObservation ---")

    game = pyhanabi.HanabiGame(game_parameters)
    print(game.parameter_string(), end="")
    obs_encoder = pyhanabi.ObservationEncoder(game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

    state = game.new_initial_state()
    num_players = game_parameters["players"]
    assert human_players <= num_players
    agents = [CustomAgent() for i in range(num_players - human_players)]

    # play until either deck is empty or total lives are lost
    while not state.is_terminal():
        # when a card must be dealt and the deck is non-empty, the state.cur_player() switches to -1
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        # fetch current agents observation
        observation = state.observation(state.cur_player())
        legal_moves = observation.legal_moves()
        print_observation(observation)

        # get input from player or agent
        if state.cur_player() in range(1, human_players + 1):
            # real player with number state.cur_player() is to move
            # wait for input from current_player
            print(f"### Possible Legal moves are: {legal_moves}")
            idx_move = input("Select Index of move to choose from:")
            # Todo: pick from UI HERE
            move = legal_moves[int(idx_move)]

        else:  # computers turn
            # Let agent compute its turn
            agents[state.cur_player()-1].act(observation)

            move = np.random.choice(legal_moves)

        state.apply_move(move)
        # Todo: Update UI based on new state

    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    run_game({"players": 3, "random_start_player": True}, 1)
