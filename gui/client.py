#!/usr/bin/python3
""" PROJECT LVL IMPORTS """
from game_state_wrapper import GameStateWrapper
import gui_config as conf, utils, commandsWebSocket as cmd
from agents.simple_agent import SimpleAgent
""" PYTHON IMPORTS """
from typing import Dict
import requests
import websocket
import threading
import time
import re
from itertools import count
import argparse


BROWSERS = [
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) snap Chromium/74.0.3729.131 Chrome/74.0.3729.131 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'
]


class Client:
    # counts how often this class has been instantiated, e.g. for determining first agent to open a lobby
    # and makes them iterable
    _ids = count(0)

    def __init__(self, url: str, cookie: str, client_config: Dict, agent_config: Dict):
        """ Client wrapped around the agents, so they can play on Zamiels server
         https://github.com/Zamiell/hanabi-live. They compute actions offline
         and send back the corresponding json-encoded action on their turn."""
        # Hanabi playing agent
        self.agent = eval(conf.AGENT_CLASSES[client_config['agent_class']]['class'])(agent_config)
        time.sleep(1)
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

        # throttle to avoid race conditions
        self.throttle = 0.05  # 50 ms

        # Agents username as seen in the server lobby
        assert 'username' in client_config
        self.username = client_config['username']
        # Stores observations for agent
        self.game = GameStateWrapper(client_config)

        # Will be set when server sends notification that a game has been created (auto-join always joins last game)
        self.gameID = None

        # Tell the Client, where in the process of joining/playing we are
        self.gottaJoinGame = False
        self.gameHasStarted = False
        self.game_ended = False

        # configuration needed for hosting a lobby
        self.config = client_config

        # current number of players in the lobby, used when our agent hosts lobby and wants to know when to start game
        self._num_players_in_lobby = -1

        # the agents will play num_episodes and then idle
        """ Note that you can watch all the replays from the server menu 'watch specific replay'.
        The ID is logged in chat"""
        self.num_episodes = self.config['num_episodes']
        self.episodes_played = 0

    def on_message(self, ws, message):
        """ Forwards messages to game state wrapper and sets flags for self.run() """
        # JOIN GAME
        # todo: print(message) if --verbose
        if message.strip().startswith('table') and not self.gameHasStarted:  # notification opened game
            self._update_latest_game_id(message)

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

    def _update_latest_game_id(self, message):
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

        # Update the latest/next game
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
            This implements the event-loop where we play Hanabi """

            # Client automatically sets gameID to the last opened game [see self.on_message()]
            if gameID is None:
                gameID = self.gameID

            # Just in case, as we sometimes get delays in the beginning (idk why)
            conn_timeout = 5
            while not self.ws.sock.connected and conn_timeout:
                time.sleep(1)
                conn_timeout -= 1

            # While client is listening
            while self.ws.sock.connected:
                # Loop to play the best game in the world :) as long as specified
                while self.episodes_played < self.num_episodes:

                    # EITHER HOST A TABLE (when 0 human players), always first client instance hosts game
                    if self.config['num_human_players'] == 0 and (self.id == 0):
                        # open a lobby
                        if not self.gameHasStarted:
                            self.ws.send(cmd.gameCreate(self.config))
                            self.gameHasStarted = True  # This is a little trick, by which we avoid rejoining own lobby
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
                            # wait to feel more human :D
                            time.sleep(self.config['wait_move'])
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
                            self.episodes_played += 1

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
        'User-Agent'] = BROWSERS[num_client]  # to avoid caching errors we immitate different BROWSERS
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

    # to play on Zamiels server or private subnets
    if args.remote_address is not None:
        addr = str(args.remote_address)
    else:
        addr = "localhost"

    # todo check if addr is valid
    referer = 'http://' + addr + '/'
    addr_ws = 'http://' + addr + '/ws'

    return addr, referer, addr_ws


def get_agent_name_from_cls(agent_class: str, id: int):
    """ Input: agentclass as specified in the conf.AGENT_CLASSES """
    assert agent_class in conf.AGENT_CLASSES
    assert 'class' in conf.AGENT_CLASSES[agent_class]

    return conf.AGENT_CLASSES[agent_class]['class'] + '0' + str(id)


def parse_variant(game_variant: str, players: int) -> Dict:
    """ Takes game variant string as required by UI server and returns game_config Dict as required by pyhanabi"""
    if game_variant == 'No Variant':
        """ Game config as required by pyhanabi.HanabiGame. """
        game_config = {

            'colors': 5,  # Number of colors in [1,5]
            'ranks': 5,  # Number of ranks in [1,5]
            'players': players,  # Number of total players in [2,5]
            'max_information_tokens': 8,  # >= 0
            'max_life_tokens': 3,  # >= 1
            'observation_type': 1,  # 0: Minimal observation, 1: First-order common knowledge obs,
            'seed': -1,  # -1 to use system random device to get seed
            'random_start_player': True  # If true, start with random player, not 0
        }
    else:
        raise NotImplementedError

    return game_config


def get_game_config_from_args(cmd_args) -> Dict:

    # total number of players ingame
    players = int(cmd_args.num_humans) + len(cmd_args.agent_classes)

    # pyhanabi-like game_config
    game_config = parse_variant(cmd_args.game_variant, players)

    return game_config


def get_client_config_from_args(cmd_args, game_config, agent: int) -> Dict:
    players = cmd_args.num_humans + len(cmd_args.agent_classes)
    deck_size = (game_config['colors'] * 2 + 1) if game_config['ranks'] < 5 else (game_config['colors'] * 10)
    client_config = {
        'agent_class': cmd_args.agent_classes[agent],
        'username': get_agent_name_from_cls(cmd_args.agent_classes[agent], agent),
        'num_human_players': cmd_args.num_humans,
        'num_total_players': players,
        "players": players,
        'empty_clues': False,
        'table_name': cmd_args.table_name,
        'table_pw': cmd_args.table_pw,
        'variant': cmd_args.game_variant,
        'num_episodes': cmd_args.num_episodes,
        'life_tokens': game_config['max_life_tokens'],
        'info_tokens': game_config['max_information_tokens'],
        'deck_size': deck_size,
        'wait_move': cmd_args.wait_move,
        'colors': game_config['colors'],
        'ranks': game_config['ranks'],
        'hand_size': utils.get_hand_size(players),
        'max_moves': utils.get_num_actions(game_config)
    }
    return client_config


def get_configs_from_args(cmd_args) -> Dict:
    """ Returns a dictionary of configurations for each agent playing. Each configuration consits of a client_confi
    and an agent_config as required by the corresponding agent class """
    # loop each agent to get the config (depending on agents class etc)
    configs = dict()

    # Compute pyhanabi game_config common for all agents
    num_agents = len(cmd_args.agent_classes)
    game_config = get_game_config_from_args(cmd_args)

    # Each agent needs config for his client instance and his agent instance
    for i in range(num_agents):

        # Config for client instance, e.g. username etc
        client_config = get_client_config_from_args(cmd_args, game_config, i)

        # Config for agent instance, e.g. num_actions for rainbow agent
        conf = utils.get_agent_config(client_config, cmd_args.agent_classes[i])

        # concatenate with game_config
        agent_config = dict(conf, **game_config)

        configs['agent'+'0'+str(i)] = {'agent_config': agent_config, 'client_config': client_config}

    return configs


def commands_valid(args):
    """ This function returns True, iff the user specified input does not break the rules of the game"""
    # assert legal number of total players
    assert 1 < args.num_humans + len(args.agent_classes) < 6
    # assert table name only contains characters, as otherwise the server will complain
    assert args.table_name.isalpha()
    # ... whatever else will come to my mind
    return True


def init_args(argparser):
    argparser.add_argument(
        '-n',
        '--num_humans',
        help='Number of human players expected at the table. Default is n=1. If n=0, the client will enter AGENTS_ONLY '
             'mode, where agents create a table for themselves and play a number of games specified with -e flag. For'
             'instance by calling "client.py -n 0 -e 1 -a simple simple", 2 simple agents will create a lobby and '
             'play for 1 round and then idle. You can watch the replay by using the "watch specific replay" option '
             'from the server with the ID of the game (that is being sent to lobby chat after game is finished).',
        type=int,
        default=1
    )
    argparser.add_argument(
        '-e',
        '--num_episodes',
        help='Number of games that will be played until agents idle. Default is e=1. -e flag will only be parsed when '
             '-n flag is set to 0, i.e. in AGENTS_ONLY mode',
        default=1
    )
    argparser.add_argument(
        '-a',
        '--agent_classes',
        help='Expects agent-class keywords as specified in client_config.py. Example: \n client.py -a simple rainbow '
             'simple \n will run a game with 2 SimpleAgent instances and 1 RainbowAgent instance. Default is simple '
             'simple, i.e. running with 2 SimpleAgent instances',
        nargs='+',
        type=str,
        default=['simple', 'simple'])
    # argparser.add_argument('-r', '--remote_address', default=None)
    argparser.add_argument(
        '-r',
        '--remote_address',
        help='Set this to an ipv4 address, when playing on a remote server or on a private subnet (with multiple '
             'humans). For example -r 192.168.178.26 when you want to connect your friends machines in a private '
             'subnet to the machine running the server at 192.168.178.26 or -r hanabi.live when you want to play on '
             'the official server. Unfortunately, it currently does not work with eduroam.',
        default='localhost'
    )
    argparser.add_argument(
        '-w',
        '--wait_move',
        help='Setting -w 2 will make each agent wait for 2 seconds before acting. Default is w=1. Is used to make the '
             'game feel more natural.',
        default=1)
    argparser.add_argument(
        '-v',
        '--verbose',
        help='Enabling --verbose, will enable verbose mode and print the whole game state instead of just the actions.',
        action='store_true')
    argparser.add_argument(
        '-t',
        '--table_name',
        help='When running the client in AGENTS_ONLY mode, i.e. when setting -n to 0, you can pass a table name with '
             'this flag. Default is "AI_room". Usually there is no need to do this though.',
        default='AIroom')
    argparser.add_argument(
        '-p',
        '--table_pw',
        help='Sets table password for AGENTS_ONLY mode, i.e. when -n is set to 0. Default password is set to '
             '"big_python", so usually you should not worry about providing -p value.',
        default='')
    argparser.add_argument(
        '-g',
        '--game_variant',
        help='Example: "Three Suits" or "No Variant". Needed for servers table creation, so make sure to '
             'not make typos here. Sets a  game_variant for the agent opening a lobby in AGENTS_ONLY mode, '
             'i.e. when -n is set to 0.',
        default='No Variant')

    args = argparser.parse_args()

    assert commands_valid(args)

    return args


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    args = init_args(argparser)

    # If agents play without human player, one of them will have to open a game lobby
    # make it passworded by default, as this will not require changes when playing remotely
    configs = get_configs_from_args(args)

    # Returns subnet ipv4 in private network and localhost otherwise
    addr, referer, addr_ws = get_addrs(args)

    clients = []
    process = []

    # Run each client in a seperate thread
    for i in range(len(args.agent_classes)):

        # get game config for current agent
        config = configs['agent'+'0'+str(i)]
        username = config['client_config']['username']
        agent_config = config['agent_config']
        client_config = config['client_config']

        # Login to Zamiels server (session-based)
        session, cookie = login(url='http://' + addr, referer=referer, username=username, num_client=i)

        # Connect the agent to websocket url
        upgrade_to_websocket(url=addr_ws, session=session, cookie=cookie)

        # Start client (agent + websocket + rl_env game wrapper)
        c = Client('ws://' + addr + '/ws', cookie, client_config, agent_config)
        clients.append(c)

        # append to lists, s.t. threads can be started later and their objects dont get removed from garbage collector
        client_thread = threading.Thread(target=c.run)
        process.append(client_thread)

    for thread in process:
        thread.start()

    # todo send gameJoin(gameID, password) when self.config['table_pw] is not '' for when -r is specified
    # todo make formatting for --verbose mode and write wiki entry for client
    # self.config['table_pw'] shall not be '' if -r is specified
