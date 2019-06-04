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
            'num_actions': get_num_actions(''),  # todo
            'num_players': get_num_players('')  # todo
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


def get_num_actions(param):
    return None


def get_num_players(param):
    return None