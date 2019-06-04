from typing import Dict
import config
import vectorizer

def get_agent_config(game_config: Dict, agent: str):
    """ Performs look-up for agent in config.AGENT_CLASSES and returns individual config. The agent config must
    always be derived from the game_config. If it cannot be computed from it, its not an OpenAI Gym compatible agent.
    New agents may be added here"""

    if agent not in config.AGENT_CLASSES:
        raise NotImplementedError

    if agent == 'rainbow':
        return {
            'observation_size': get_observation_size(game_config),
            'num_actions': get_num_actions(game_config),  # todo
            'num_players': game_config['players']  # todo
        }
    elif agent == 'simple':
        return {
            'players': game_config['players']
        }
    else:
        raise NotImplementedError


def get_observation_size(game_config):
    num_players = game_config['num_total_players']  # number of players ingame
    num_colors = game_config['colors']
    num_ranks = game_config['ranks']
    hand_size = game_config['hand_size']
    max_info_tokens = game_config['life_tokens']
    max_life_tokens = game_config['info_tokens']
    max_moves = game_config['max_moves']
    variant = game_config['variant']
    env = create_env_mock(
        num_players=num_players,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=hand_size,
        max_info_tokens=max_info_tokens,
        max_life_tokens=max_life_tokens,
        max_moves=max_moves,
        variant=variant
    )

    vec = vectorizer.ObservationVectorizer(env)
    legal_moves_vectorizer = vectorizer.LegalMovesVectorizer(env)
    return vec.total_state_length

def get_hand_size(players: int) -> int:
    """ Returns number of cards in each players hand, depending on the number of players """
    assert 1 < players < 6
    return 4 if players > 3 else 5


def get_num_actions(game_config):
    """ total number of moves possible each turn (legal or not, depending on num_players and num_cards),
    i.e. MaxDiscardMoves + MaxPlayMoves + MaxRevealColorMoves + MaxRevealRankMoves """
    hand_size = get_hand_size(game_config['players'])
    num_colors = game_config['colors']
    num_ranks = game_config['ranks']
    num_players = game_config['players']

    return 2 * hand_size + (num_players - 1) * num_colors + (num_players - 1) * num_ranks



def create_env_mock(num_players, num_colors, num_ranks, hand_size, max_info_tokens, max_life_tokens, max_moves, variant):
    num_players = num_players
    num_colors = num_colors
    num_ranks = num_ranks
    hand_size = hand_size
    max_info_tokens = max_info_tokens
    max_life_tokens = max_life_tokens
    max_moves = max_moves
    variant = variant

    return envMock(
        num_players=num_players,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=hand_size,
        max_info_tokens=max_info_tokens,
        max_life_tokens=max_life_tokens,
        max_moves=max_moves,
        variant=variant
    )

class envMock:
    def __init__(self, num_players, num_colors, num_ranks, hand_size, max_info_tokens, max_life_tokens, max_moves, variant):
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.max_info_tokens = max_info_tokens
        self.max_life_tokens = max_life_tokens
        self.max_moves = max_moves
        self.variant = variant

    def num_cards(self, color, rank, variant):
        """ Input: Color string in "RYGWB" and rank in [0,4]
        Output: How often deck contains card with given color and rank, i.e. 1-cards will be return 3"""
        if rank == 0:
            return 3
        elif rank < 4:
            return 2
        elif rank == 4:
            return 1
        else:
            return 0