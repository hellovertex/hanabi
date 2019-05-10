#!/usr/bin/python3
import typing

def hello() -> str:
    """ Upon joining game table """
    return '{"hello":{}}'

def gameJoin(gameID: str) -> str:
    """ To join game table from lobby """
    return 'gameJoin {"gameID":' + gameID + '}'

def ready() -> str:
    """ After hello() upon joining game table """
    return '{"ready":{}}'
