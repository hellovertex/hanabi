#!/usr/bin/python3
import typing
from typing import Dict
import json
import ast

def hello() -> str:
    """ Upon joining game table """
    return 'hello {}'


def gameJoin(gameID: str) -> str:
    """ To join game table from lobby """
    return 'gameJoin {"gameID":' + gameID + '}'


def ready() -> str:
    """ After hello() upon joining game table """
    return 'ready {}'


def gameUnattend():
    return 'gameUnattend {}'


def _get_table_params(config: Dict) -> Dict:
    """ This method is called when no human player is playing and an AI agent hosts a lobby.
    In order to open a game lobby he needs to send a json encoded lobby config, for which we get the params here."""

    # if they decide to change the lobby parameters on the server code some day, we have a problem when we want
    # to let our agents play remotely on Zamiels server, lol. But lets assume that this is very unlikely to happen.
    game_config = dict()
    game_config['name'] = config['table_name']
    game_config['variant'] = config['variant']
    game_config['timed'] = 'false'
    game_config['baseTime'] = 120
    game_config['timePerTurn'] = 20
    game_config['speedrun'] = 'false'
    game_config['deckPlays'] = 'false'
    game_config['emptyClues'] = str(config['empty_clues']).lower()  # parse bool flag to str
    game_config['characterAssignments'] = 'false'
    game_config['correspondence'] = 'false'
    game_config['password'] = config['table_pw']
    game_config['alertWaiters'] = 'false'

    return game_config


def gameCreate(config: Dict):
    lobby_config = _get_table_params(config)
    return 'gameCreate ' + json.dumps(lobby_config).replace('"false"', 'false').replace(''"true"'', 'true')


def gameStart():
    return 'gameStart {}'


def dict_from_response(response: str, msg_type: str=None) -> Dict:
    assert msg_type is not None
    d = ast.literal_eval(
        response.split(msg_type.strip()+' ')[1].replace('false', 'False').replace('list', 'List').replace('true', 'True')
    )
    return d
""" ACTIONS INGAME
clue: { // Not present if the type is 1 or 2
			type: 0, // 0 is a rank clue, 1 is a color clue
			value: 1, // If a rank clue, corresponds to the number
			// If a color clue:
			// 0 is blue
			// 1 is green
			// 2 is yellow
			// 3 is red
			// 4 is purple
			// (these mappings change in the mixed variants)
		},
"""

"""{'current_player': 0,
                                  'current_player_offset': 0,
                                  'deck_size': 40,
                                  'discard_pile': [],
                                  'fireworks': {'B': 0,
                                                'G': 0,
                                                'R': 0,
                                                'W': 0,
                                                'Y': 0},
                                  'information_tokens': 8,
                                  'legal_moves': [{'action_type': 'PLAY',
                                                   'card_index': 0},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 1},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 2},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 3},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 4},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'R',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'G',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'B',
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 0,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 1,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 2,
                                                   'target_offset': 1}],
                                  'life_tokens': 3,
                                  'observed_hands': [[{'color': None, 'rank':
                                  -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1}],
                                                     [{'color': 'G', 'rank': 2},
                                                      {'color': 'R', 'rank': 0},
                                                      {'color': 'R', 'rank': 1},
                                                      {'color': 'B', 'rank': 0},
                                                      {'color': 'R', 'rank':
                                                      1}]],
                                  'num_players': 2,
                                  'vectorized': [ 0, 0, 1, ... ]},
                                 {'current_player': 0,
                                  'current_player_offset': 1,
                                  'deck_size': 40,
                                  'discard_pile': [],
                                  'fireworks': {'B': 0,
                                                'G': 0,
                                                'R': 0,
                                                'W': 0,
                                                'Y': 0},
                                  'information_tokens': 8,
                                  'legal_moves': [],
                                  'life_tokens': 3,
                                  'observed_hands': [[{'color': None, 'rank':
                                  -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1}],
                                                     [{'color': 'W', 'rank': 2},
                                                      {'color': 'Y', 'rank': 4},
                                                      {'color': 'Y', 'rank': 2},
                                                      {'color': 'G', 'rank': 0},
                                                      {'color': 'W', 'rank':
                                                      1}]],
                                  'num_players': 2,
                                  'vectorized': [ 0, 0, 1, ... ]}"""

def action(type: int, target: int, clue: Dict[str, str]) -> str:

    cluetype = str(clue["type"])
    cluevalue = str(clue["value"])

    return 'action {"type":'+type+',"target":'+target+'"clue":{"type":'+cluetype+',"value:":'+cluevalue+'}}'

