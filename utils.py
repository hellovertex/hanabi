from typing import Dict
import config


def get_agent_config(game_config: Dict, agent: str):
    """ Performs look-up for agent in config.AGENT_CLASSES and returns individual config. The agent config must
    always be derived from the game_config. If it cannot be computed from it, its not an OpenAI Gym compatible agent.
    New agents may be added here"""

    if agent not in config.AGENT_CLASSES:
        raise NotImplementedError

    if agent == 'rainbow':
        return {
            'observation_size': get_hand_size(game_config['players']),
            'num_actions': get_num_actions(game_config),  # todo
            'num_players': game_config['players']  # todo
        }
    elif agent == 'simple':
        return {
            'players': game_config['players']
        }
    else:
        raise NotImplementedError


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



