#!/usr/bin/python3
""" PROJECT LVL IMPORTS """
import commandsWebSocket as cmd
from game_state_wrapper import GameStateWrapper
from agents.simple_agent import SimpleAgent
import client_config as conf

""" PYTHON IMPORTS """
from typing import Dict
import requests
import websocket
import threading
import time
import re
import sys
from itertools import count
import argparse


browsers = [
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) snap Chromium/74.0.3729.131 Chrome/74.0.3729.131 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'
]


class Client:
    # counts how often this class has been instantiated, e.g. for determining first agent to open a lobby
    # and makes them iterable
    _ids = count(0)

    def __init__(self, url: str, cookie: str, game_config: Dict, agent_config: Dict):
        """ Client wrapped around the agents, so they can play on Zamiels server
         https://github.com/Zamiell/hanabi-live. They compute actions offline
         and send back the corresponding json-encoded action on their turn."""

        # Opens a websocket on url:80
        self.ws = websocket.WebSocketApp(url=url,
                                         on_message=lambda ws, msg: self.on_message(ws, msg),
                                         on_error=lambda ws, msg: self.on_error(ws, msg),
                                         on_close=lambda ws, msg: self.on_close(ws),
                                         cookie=cookie)

        # Set on_open seperately as it does crazy things otherwise #pythonWebsockets
        self.ws.on_open = lambda ws: self.on_open(ws)

        # listen for incoming messages
        self.daemon = threading.Thread(target=self.ws.run_forever)
        self.daemon.daemon = True
        self.daemon.start()  # [do this before self.run(), s.t. we can hand over the daemon to another Thread]

        # increment class instance counter
        self.id = next(self._ids)

        # Hanabi playing agent
        self.agent = SimpleAgent(agent_config)

        # Store incoming server messages here, to get observations etc.
        self.msg_buf = list()  # maybe replace this with a game_state object

        # throttle to avoid race conditions
        self.throttle = 0.05  # 50 ms

        # Stores observations for each agent
        self.username = game_config['username']
        self.game = GameStateWrapper(agent_config['players'], self.username)

        # Will always be set to the game created last (on the server side ofc)
        self.gameID = None

        # Tell the Client, where in the process of joining/playing we are
        self.gottaJoinGame = False
        self.gameHasStarted = False
        self.game_ended = False

        # configuration needed for hosting a lobby
        self.config = game_config

        # current number of players in the lobby, used when our agent hosts lobby and wants to know when to start game
        self._num_players_in_lobby = -1
        self.reset_interrupted_games = self.config['ff']

    def on_message(self, ws, message):
        """ Forwards messages to game state wrapper and sets flags for self.run() """
        # JOIN GAME
        # todo: print(message) if --verbose
        if message.strip().startswith('table') and not self.gameHasStarted:  # notification opened game
            self._set_auto_join_game(message)

        # HOSTED TABLE
        if message.startswith('game {') and self.id == 0:  # always first agent to host a table
            self._update_num_players_in_lobby(message)

        # START GAME
        if message.strip().startswith('gameStart'):
            self.ws.send(cmd.hello())  # ACK GAME START

        # INIT GAME
        if message.strip().startswith('init'):
            self.ws.send(cmd.ready())  # ACK GAME INIT
            self.game.init_players(message)  # set list of players
            self.gameHasStarted = True

        # CARDS DEALT
        if message.startswith('notifyList '):
            self.game.deal_cards(message)

        # UPDATE GAME STATE
        if message.startswith('notify '):
            self.game.update_state(message)

        # END GAME
        if message.startswith('gameOver'):
            self.game_ended = True

    def _set_auto_join_game(self, message):
        """ Set joingame-flags for self.run(), s.t. it joins created games whenever possible"""
        # To play another game after one is finished
        oldGameID = None

        # If no game has been created yet, we will join the next one
        if self.gameID is None:
            self.gottaJoinGame = True
        else:
            oldGameID = self.gameID

        # get current latest opened game lobby id
        tmp = message.split('id":')[1]
        self.gameID = str(re.search(r'\d+', tmp).group())

        # Join the latest/next game
        if oldGameID is not None and self.gameID > oldGameID:
            self.gottaJoinGame = True

    def _update_num_players_in_lobby(self, message: str):
        # parse dict
        response = cmd.dict_from_response(message, msg_type='game')

        # update count of players in lobby
        if 'players' in response:
            self._num_players_in_lobby = len(response['players'])

    @staticmethod
    def on_error(ws, error):
        print("Error: ", error)

    @staticmethod
    def on_close(ws):
        print("### closed ###")

    @staticmethod
    def on_open(ws):
        """ Zamiels server doesnt require any hello-messages"""
        pass

    def run(self, gameID=None):
            """ Basically the main workhorse.
            This implements the event-loop where we process incoming and outgoing messages """

            # Client automatically sets gameID to the last opened game [so best only open one at a time]
            if gameID is None:
                gameID = self.gameID

            # Just in case, as we sometimes get delays in the beginning (idk why)
            conn_timeout = 5
            while not self.ws.sock.connected and conn_timeout:
                time.sleep(1)
                conn_timeout -= 1

            # Loop to play the best game in the world :)
            while self.ws.sock.connected:
                if self.reset_interrupted_games:
                    self.ws.send(cmd.gameUnattend())
                    self.reset_interrupted_games = False

                # EITHER HOST (when 0 human players), always first client instance hosts game
                if self.config['num_human_players'] == 0 and (self.id == 0):
                    # open a lobby
                    if not self.gameHasStarted:
                        self.ws.send(cmd.gameCreate(self.config))
                        print("GAME CREATED")
                        self.gameHasStarted = True  # This is a little trick, by which we avoid rejoining our lobby
                        # nothing will happen in the run() method,
                        # as all others involved flags are still set to False

                    # when all players have joined, start the game
                    if self._num_players_in_lobby == self.config['num_total_players']:
                        self.ws.send(cmd.gameStart())

                    time.sleep(1)

                # OR JOIN GAME
                elif self.gottaJoinGame and self.gameID:
                    time.sleep(self.throttle)
                    self.ws.send(cmd.gameJoin(gameID=self.gameID))
                    self.gottaJoinGame = False

                # PLAY GAME
                if self.gameHasStarted:  # set in self.on_message() on servers init message

                    # ON AGENTS TURN
                    if self.game.agents_turn:
                        # wait a second, to feel more human :D
                        time.sleep(1)
                        # Get observation
                        obs = self.game.get_agent_observation()
                        # Compute action
                        a = self.agent.act(obs)
                        # Send to server
                        self.ws.send(self.game.parse_action_to_msg(a))

                    # leave replay lobby when game has ended
                    if self.game_ended:
                        self.ws.send(cmd.gameUnattend())
                        self.game_ended = False
                        self.gameHasStarted = False

                time.sleep(1)


class ProgressThread(threading.Thread):
    """ Mainly for debugging. Can be removed later"""
    def __init__(self, client_daemon, ws):
        super(ProgressThread, self).__init__()

        self.worker = client_daemon
        self.ws = ws

    def run(self):
        while True:
            if not self.worker.is_alive():
                'Client has disconnected'
                return True
            print("thread2")
            print(self.ws.msg_buf)
            time.sleep(1.0)


def login(url, referer, username,
          password="01c77cf03d35866f8486452d09c067f538848058f12d8f005af1036740cccf98", num_client=0):
    """ Make POST request to /login page"""
    ses = requests.session()
    # set up header
    # set useragent to chrome, s.t. we dont have to deal with the additional firefox warning
    ses.headers[
        'User-Agent'] = browsers[num_client]  # to avoid caching errors we immitate different browsers
    ses.headers['Accept'] = '*/*'
    ses.headers['Accept-Encoding'] = 'gzip, deflate'
    ses.headers['Accept-Language'] = 'en-US,en;q=0.5'
    ses.headers['Origin'] = url

    url = url + '/login'
    data = {
        "username": username,
        "password": password
    }
    headers = {
        # "Accept": "*/*",
        # "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        # "Origin": url,
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest"
    }
    # attempt login and save HTTP response
    response = ses.post(url=url, data=data, headers=headers)

    # store cookie because we need them for web socket establishment
    cookie = response.headers['set-cookie']

    assert response.status_code == 200  # assert login was successful

    return ses, cookie


def upgrade_to_websocket(url, session, cookie):
    """ Make a GET request to UPGRADE HTTP to Websocket"""

    # headers
    headers = {
        "Connection": "Upgrade",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Upgrade": "websocket",
        "Sec-Websocket-Version": "13",
        "Cookie": cookie,
        "Sec-WebSocket-Key": "6R/+PZjR24OnAaKVwAnTRA==",  # Base64 encoded
        "Sec-Websocket-Extensions": "permessage-deflate; client_max_window_bits"
    }
    # make get
    response = session.get(url=url, headers=headers)
    # print(f"STATUSCODE WS: {response.status_code}")

    assert response.status_code == 101  # 101 means SWITCHING_PROTOCOL, i.e. success

    return


def get_addrs(args):
    """ We will use this in private networks, to avoid localhost-related bugs for now.
    However, we have to get the localhost-settings running at some point."""

    addr = None
    referer = ''

    # to play on Zamiels server or private subnets
    if args.remote_address is not None:
        # todo check if addr is valid
        addr = str(args.remote_address)
        referer = 'http://'+addr+'/'

    # to play on localhost
    else:
        addr = "localhost"
        referer = "http://localhost/"
        #addr = '192.168.178.26'
        #referer = "http://192.168.178.26/"
    return addr, referer


def get_agent_name_from_cls(agent_class: str, id: int):
    """ Input: agentclass as specified in the client_config.py """
    assert agent_class in conf.agent_classes
    assert 'class' in conf.agent_classes[agent_class]

    return conf.agent_classes[agent_class]['class'] + '0' + str(id)


def get_game_configs_from_args(cmd_args) -> Dict:
    """ Returns a dictionary of configurations for each agent playing """
    # loop each agent to get the config (depending on agents class etc)
    configs = dict()
    num_agents = len(cmd_args.agent_classes)
    for i in range(num_agents):
        configs['agent'+'0'+str(i)] = {
        'agent_class': cmd_args.agent_classes[i],
        'username': get_agent_name_from_cls(cmd_args.agent_classes[i], i),
        'num_total_players': cmd_args.num_humans + num_agents,
        'num_human_players': cmd_args.num_humans,
        'empty_clues': False,
        'table_name': 'default',
        'table_pw': '',
        'variant': 'Three Suits',
        'num_episodes': cmd_args.num_episodes,
        'ff': cmd_args.ff
    }

    return configs


def commands_valid(args):
    """ This function returns True, iff the user specified input does not break the rules of the game"""
    # assert legal number of total players
    assert 1 < args.num_humans + len(args.agent_classes) < 6

    return True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--num_humans', default=1)
    argparser.add_argument('-e', '--num_episodes', default=1)
    argparser.add_argument('-a', '--agent_classes', nargs='+', type=str, default=['simple', 'simple'])
    argparser.add_argument('-r', '--remote_address', default='192.168.178.26')
    argparser.add_argument('--ff', action='store_false')

    args = argparser.parse_args()
    assert commands_valid(args)

    # If agents play without human player, one of them will have to open a game lobby
    # make it passworded by default, as this will not require changes when playing remotely
    agent_config = {'players': args.num_humans + 2}
    game_configs = get_game_configs_from_args(args)

    # Returns subnet ipv4 in private network and localhost otherwise
    addr, referer = get_addrs(args)

    # Login to Zamiels server (session-based)
    session, cookie = login(url='http://' + addr, username=game_configs['agent00']['username'], referer=referer)
    upgrade_to_websocket(url='http://' + addr + '/ws', session=session, cookie=cookie)

    # Connect the agent to websocket url
    url = 'ws://' + addr + '/ws'


    # process = []
    # for i in range(len(agentclasses)):
    #     print(agentclasses[i], type(agentclasses[i]))
    #     username = agentclasses[i]+'0'+str(i)
    #     session, cookie = login(url='http://' + addr, referer=referer, username=username, num_client=i)
    #     upgrade_to_websocket(url='http://' + addr + '/ws', session=session, cookie=cookie)
    #
    #     # Create one thread per agent
    #     c = Client(url, cookie, username, config)
    #     client_thread = threading.Thread(target=c.run)
    #     client_thread.start()
    #     process.append(client_thread)
    #
    # for thread in process:
    #     thread.join()

    c1 = Client(url, cookie, game_configs['agent00'], agent_config)
    c1_thread = threading.Thread(target=c1.run)

    c1_thread.start()

    session, cookie = login(url='http://' + addr, referer=referer, username=game_configs['agent01']['username'],
                            num_client=1)
    upgrade_to_websocket(url='http://' + addr + '/ws', session=session, cookie=cookie)

    c2 = Client(url, cookie, game_configs['agent01'], agent_config)
    c2_thread = threading.Thread(target=c2.run)
    c2_thread.start()

    # TODO s: implement argparser, create config and generate usernames, implement threaded clients
    # TODO s: restarting -n -a -r  -l -p - w- v- g -e
    # if n==0 then agent creates game
    # 1. agent creating lobby

