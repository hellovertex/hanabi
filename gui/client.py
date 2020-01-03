#!/usr/bin/python3
""" PROJECT LVL IMPORTS """
from server_game import GameStateWrapper
from gui_agent import AGENT_CLASSES, RainbowAgent, PPOAgent, SimpleAgent
import gui_utils
import json_to_websocket as json_utils
""" PYTHON IMPORTS """
from typing import Dict
import requests
import websocket
import threading
import enum
import time
import re
from itertools import count
import argparse
import os
import sys

# Just to immitate clients that have compatible browsers
BROWSERS = [
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) snap Chromium/74.0.3729.131 Chrome/74.0.3729.131 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36']
# Each AI agent has its own "account" stored in the database, using the following password
PWD = "01c77cf03d35866f8486452d09c067f538848058f12d8f005af1036740cccf98"
# disable tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information


class ClientMode(enum.IntEnum):
    # number of human players
    AGENTS_ONLY = 0
    WITH_ONE_HUMAN_PLAYER = 1
    WITH_TWO_HUMAN_PLAYERS = 2
    WITH_THREE_HUMAN_PLAYERS = 3
    WITH_FOUR_HUMAN_PLAYERS = 4


class Client:
    """ Client wrapped around the instances of gui_agents.GUIAgents, so they can play on Zamiels server
         https://github.com/Zamiell/hanabi-live. They compute actions offline
         (assuming pyhanabi observation-dict with canonical encoding scheme for vectorized observation)
         and send back the corresponding json-encoded action on their turn."""
    # counts how often this class has been instantiated, e.g. for determining first agent who hosts a lobby
    # and makes them iterable
    _ids = count(0)

    def __init__(self, url: str, cookie: str, username, cmd_args):
        # Hanabi playing agent
        # note that the used Agent class must be imported here from gui_agent.py
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
        self.id = next(self._ids)  # first call to next(self._ids) will return 0

        # throttle to avoid race conditions
        self.throttle = 0.05  # 50 ms

        # Agents username as seen in the server lobby
        self.username = username

        """ 
        AGENTS_ONLY: agents config is derived exclusively from the command line arguments passed
        WITH_HUMAN: agents config is derived from the params of the game create by human via GUI
        """
        client_mode = cmd_args.subparser_name
        assert client_mode in ['agents_only', 'with_human']
        # In agents_only-mode the agents will host a game with given arguments
        # Otherwise, a human will set up the game lobby
        if client_mode == 'agents_only':
            self.mode = ClientMode.AGENTS_ONLY
        # Else, mode will be equal to the number of human players
        else:
            assert hasattr(cmd_args, 'num_humans')
            self.mode = cmd_args.num_humans

        # configuration needed for hosting/joining a lobby
        self.config = self.load_client_config(cmd_args)

        # instantiated later
        self.game = None
        self.agent_class = AGENT_CLASSES[cmd_args.agent_classes[self.id]]

        # In agents only mode, we can instantiate the agents immediately
        if self.mode == ClientMode.AGENTS_ONLY:
            # In agents only mode, the pyhanabi_config can be determined from cmd_args
            pyhanabi_config = self.load_pyhanabi_config(cmd_args)
            # and we can load config for agent class using its static classmethod load_config()
            agent_config = eval(self.agent_class).load_config(pyhanabi_config)
            # to init agent <gui_agent.GUIAgent subclass>
            self.agent = eval(self.agent_class)(agent_config)
            # and game
            self.game = GameStateWrapper(self.load_game_config(pyhanabi_config))
        # Otherwise, we have to wait for the human player to create a game
        else:
            self.agent = None
            self.game = None

        # set agent admin for hosting game, if playing without humans
        if self.id == 0 and self.mode == ClientMode.AGENTS_ONLY:
            self.game.caller_is_admin = True

        # Will be set when server sends notification that a game has been created
        # (auto-join will always join latest game)
        self.gameID = None

        # Tell the Client, where in the process of joining/playing we are
        self.gottaJoinGame = False
        self.gameHasStarted = False
        self.game_created = False
        self.game_paused = False

        # current number of players in the lobby, used when our agent hosts lobby and wants to know when to start game
        self._num_players_in_lobby = -1


        self.episodes_played = 0
        self.cmd_args = cmd_args

    def load_client_config(self, args):
        if self.mode == ClientMode.AGENTS_ONLY:
            table_name = args.table_name
            table_pw = args.table_pw
            variant = gui_utils.variant_from_num_colors(args.num_colors)
            num_episodes = args.num_episodes
        else:  # table will be opened by human
            table_name = ''
            table_pw = ''
            variant = ''
            num_episodes = float("inf")
        num_agents = len(args.agent_classes)
        num_total_players = num_agents if self.mode == ClientMode.AGENTS_ONLY else num_agents + args.num_humans
        return {'table_name': table_name,
                'table_pw': table_pw,
                'variant': variant,
                'num_human_players': self.mode,
                'num_total_players': num_total_players,
                'num_episodes': num_episodes,
                'agent_class': args.agent_classes[self.id],
                'username': self.username,
                'wait_move': 1}

    def load_pyhanabi_config(self, cmd_args, table_params=None):
        assert hasattr(cmd_args, 'subparser_name')
        # can only get the pyhanabi config right from the command line arguments, if they specify
        # the number of colors, ranks,... which is only the case for the agents_only mode
        # with_human mode will not require these arguments, as they will be specified via the GUI by a human
        assert cmd_args.subparser_name in ['agents_only', 'with_human']
        if cmd_args.subparser_name == 'agents_only':
            colors = cmd_args.num_colors
        elif cmd_args.subparser_name == 'with_human':
            assert isinstance(table_params, dict)
            game_variant = table_params['variant']
            colors = gui_utils.num_colors_from_variant(game_variant)
        else:
            raise ValueError

        return {
            'colors': colors,  # Number of colors in [1,5]
            'ranks': 5,  # todo: can not change under the current server implementation
            'players': self.config['num_total_players'],  # Number of total players in [2,5]
            'hand_size': 4 if len(cmd_args.agent_classes) > 3 else 5,  # todo: can not change under the current server impl
            'max_information_tokens': cmd_args.info_tokens,  # >= 0
            'max_life_tokens': cmd_args.life_tokens,  # >= 1
            'observation_type': cmd_args.observation_type,  # 0: Minimal observation, 1: First-order common knowledge obs,
            'seed': -1,  # -1 to use system random device to get seed
            'random_start_player': False  # todo: check proper mapping from server indices
        }

    def load_game_config(self, pyhanabi_config):
        colors = pyhanabi_config['colors']
        ranks = pyhanabi_config['ranks']
        return {'username': self.username,
                'num_total_players': self.config['num_total_players'],
                'life_tokens': pyhanabi_config['max_life_tokens'],
                'info_tokens': pyhanabi_config['max_information_tokens'],
                'colors': colors,
                'ranks': ranks,
                'hand_size': pyhanabi_config['hand_size'],
                'deck_size': sum([3,2,2,2,1][:ranks]) * colors,
                'max_moves': gui_utils.get_num_actions(pyhanabi_config)}

    def quit_game(self):
        self.ws.send(json_utils.gameAbandon())
        time.sleep(self.throttle)
        # self.ws.send(json_utils.gameUnattend())
        try:
            self.ws.close()
        except Exception:
            print('close')
        finally:
            self.gameID = None
            return

        #time.sleep(self.throttle)
        #self.ws.send(json_utils.gameAbandon())

    def on_message(self, ws, message):
        """ Forwards messages to game state wrapper and sets flags for self.run() """
        # NOTIFICATION GAME OPENED
        if message.strip().startswith('table') and not self.gameHasStarted:
            # if playing with humans
            # when the game lobby is created,
            # we can initialize the agents with the derived pyhanabi configuration
            # and update the config for hosting/joining lobby
            table_params = json_utils.dict_from_response(response=message, msg_type='table')
            if not self.mode == ClientMode.AGENTS_ONLY:
                # update config
                self.config['table_name'] = table_params['name']
                self.config['table_pw'] = ''
                self.config['variant'] = table_params['variant']

                # load agent with derived pyhanabi config
                pyhanabi_config = self.load_pyhanabi_config(cmd_args=self.cmd_args, table_params=table_params)
                agent_config = eval(self.agent_class).load_config(pyhanabi_config)
                self.agent = eval(self.agent_class)(agent_config)

                # create game object that will return agents_observations
                game_config = self.load_game_config(pyhanabi_config)
                self.game = GameStateWrapper(game_config)  # Stores observations for agent
            # self._maybe_abandon_old_unfinished_game(table_params)
            self._update_latest_game_id_and_set_join_flag(message)

        # REFRESH CURRENT TABLE
        if message.startswith('game {') and self.id == 0:  # always first agent to host a table
            self._update_num_players_in_lobby(message)

        # START GAME
        if message.strip().startswith('gameStart'):
            self.ws.send(json_utils.hello())  # ACK GAME START

        # INIT GAME
        if message.strip().startswith('init'):
            self.ws.send(json_utils.ready())  # ACK GAME INIT
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
            self.game.finished = True

    def _update_latest_game_id_and_set_join_flag(self, message):
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
        response = json_utils.dict_from_response(message, msg_type='game')

        # update count of players in lobby
        if 'players' in response:
            self._num_players_in_lobby = len(response['players'])

    def _maybe_abandon_old_unfinished_game(self, table_params):
        if self.username in table_params['players']:
            # there is a hanging game, which we must leave to be able to join a new one
            self.quit_game()

    @staticmethod
    def on_error(ws, error):
        print("Error: ", error)

    @staticmethod
    def on_close(ws):
        pass

    @staticmethod
    def on_open(ws):
        """ Zamiells server doesnt require any hello-messages"""
        pass

    def run(self, gameID=None):
            """ Implements the event-loop where Hanabi is played """

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
                # The agents will try to play num_episodes and then disconnect
                while self.episodes_played < self.config['num_episodes']:

                    # EITHER HOST A TABLE (when 0 human players), always first client instance hosts game
                    if self.config['num_human_players'] == 0 and (self.id == 0):
                        # open a lobby
                        if not self.gameHasStarted:
                            if not self.game_created:
                                self.ws.send(json_utils.gameCreate(self.config))
                                self.gottaJoinGame = False  # This is a little trick, by which we avoid rejoining own lobby
                                # nothing will happen in the run() method,
                                # as all others involved flags are still set to False
                                self.game_created = True
                            # when all players have joined, start the game
                        if self._num_players_in_lobby == self.config['num_total_players']:
                                self.ws.send(json_utils.gameStart())

                        time.sleep(1)

                    # OR JOIN GAME
                    elif self.gottaJoinGame and self.gameID:
                        self.ws.send(json_utils.gameJoin(gameID=self.gameID))
                        time.sleep(self.throttle)
                        self.ws.send(json_utils.gameJoin(gameID=self.gameID))
                        self.gottaJoinGame = False

                    # PLAY GAME
                    if self.gameHasStarted:  # set in self.on_message() on servers init message

                        # ON AGENTS TURN
                        if self.game.agents_turn and not self.game_paused:
                            # wait to feel more human :D
                            time.sleep(self.config['wait_move'])
                            # Get observation
                            obs = self.game.get_agent_observation()
                            # Compute action
                            a = self.agent.act(obs)
                            # Send to server
                            self.ws.send(self.game.parse_action_to_msg(a))

                        # leave replay lobby when game has ended
                        if self.game.finished:
                            self.quit_game()
                            self.game.finished = False
                            self.gameHasStarted = False
                            self.episodes_played += 1

                    time.sleep(.25)
                time.sleep(.25)


def login(url, referer, username, password=PWD, num_client=0):
    """ Make POST request to /login page"""
    sess = requests.session()
    # set up header
    # set useragent to chrome, s.t. we dont have to deal with the additional firefox warning
    sess.headers[
        'User-Agent'] = BROWSERS[num_client]  # to avoid caching errors we immitate different BROWSERS
    sess.headers['Accept'] = '*/*'
    sess.headers['Accept-Encoding'] = 'gzip, deflate'
    sess.headers['Accept-Language'] = 'en-US,en;q=0.5'
    sess.headers['Origin'] = url

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
    response = sess.post(url=url, data=data, headers=headers)

    # store cookie because we need them for web socket establishment
    cookie = response.headers['set-cookie']

    assert response.status_code == 200  # assert login was successful

    return sess, cookie


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
    """ If no remote address is provided, the server is assumed to be running on localhost """

    # to play on Zamiels server or private subnets
    if args.remote_address:
        addr = str(args.remote_address)
    else:
        addr = "localhost"

    # todo check if addr is valid
    referer = 'http://' + addr + '/'
    addr_ws = 'http://' + addr + '/ws'

    return addr, referer, addr_ws


def init_args():
    # todo share arguments across subparsers to avoid repetitive code here
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest='subparser_name')

    a_parser = subparsers.add_parser('agents_only')
    b_parser = subparsers.add_parser('with_human')

    def _add_args():
        # todo figure out how to share arguments (inherit from parent parser ?)
        # ClientMode.AGENTS_ONLY
        a_parser.add_argument(
            '-c',
            '--num_colors',
            choices=[3, 4, 5],  # server does not allow less than 3 colors
            help='Number of colors used in the game. '
                 'If not -n=ClientMode.AGENTS_ONLY, this will be ignored, as human hosts game.\n c.f.'
                 'https://github.com/Zamiell/hanabi-live/blob/master/docs/VARIANTS.md',
            default=gui_utils.GameVariants.FIVE_SUITS)
        a_parser.add_argument(
            '-i',
            '--info_tokens',
            help='Number of info_tokens used in the game. '
                 'If not -n=ClientMode.AGENTS_ONLY, this will be ignored, as human hosts game.',
            default=8)
        a_parser.add_argument(
            '-l',
            '--life_tokens',
            help='Number of life_tokens used in the game. '
                 'If not -n=ClientMode.AGENTS_ONLY, this will be ignored, as human hosts game.',
            default=3)
        a_parser.add_argument(
            '-o',  # 0: Minimal observation, 1: First-order common knowledge obs, 2: SEER
            '--observation_type',
            choices=[0, 1, 2],
            type=int,
            default=1
        )
        a_parser.add_argument(
            '-a',
            '--agent_classes',
            nargs='+',
            type=str,
            required=True,
            default=['simple', 'simple']
        )
        a_parser.add_argument(
            '-e',
            '--num_episodes',
            help='Number of games that will be played until agents idle. Default is e=1. -e flag will only be parsed when '
                 '-n flag is set to 0, i.e. in AGENTS_ONLY mode',
            default=1,
            type=int
        )
        a_parser.add_argument(
            '-t',
            '--table_name',
            help='When running the client in AGENTS_ONLY mode, i.e. when setting -n to 0, you can pass a table name with '
                 'this flag. Default is "AI_room". Usually there is no need to do this though.',
            default='AIroom')
        a_parser.add_argument(
            '-p',
            '--table_pw',
            help='Sets table password for AGENTS_ONLY mode, i.e. when -n is set to 0. Default password is set to '
                 '"big_python", so usually you should not worry about providing -p value.',
            default='')
        a_parser.add_argument(
            '-r',
            '--remote_address',
            help='Set this to an ipv4 address, when playing on a remote server or on a private subnet (with multiple '
                 'humans). For example -r 192.168.178.26 when you want to connect your friends machines in a private '
                 'subnet to the machine running the server at 192.168.178.26 or -r hanabi.live when you want to play on '
                 'the official server. Unfortunately, it currently does not work with eduroam.',
        )

        # ClientMode.WITH_HUMAN
        b_parser.add_argument(
            '-n',
            '--num_humans',
            choices=[1, 2, 3, 4],
            help='Number of human players expected at the table. Default is n=1. If n=0, the client will enter AGENTS_ONLY '
                 'mode, where agents create a table for themselves and play a number of games specified with -e flag. For'
                 'instance by calling "client.py -n=0 -e=1 -a simple simple", 2 simple agents will create a lobby and '
                 'play for 1 round and then idle. You can watch the replay by using the "watch specific replay" option '
                 'from the server with the ID of the game (that is being sent to lobby chat after game is finished).',
            type=int,
            default=ClientMode.WITH_ONE_HUMAN_PLAYER
        )
        b_parser.add_argument(
            '-a',
            '--agent_classes',
            nargs='+',
            type=str,
            required=True,  # default=['simple', 'simple'],
        )
        b_parser.add_argument(
            '-i',
            '--info_tokens',
            help='Number of info_tokens used in the game. '
                 'If not -n=ClientMode.AGENTS_ONLY, this will be ignored, as human hosts game.',
            default=8)
        b_parser.add_argument(
            '-l',
            '--life_tokens',
            help='Number of life_tokens used in the game. '
                 'If not -n=ClientMode.AGENTS_ONLY, this will be ignored, as human hosts game.',
            default=3)
        b_parser.add_argument(
            '-o',  # 0: Minimal observation, 1: First-order common knowledge obs, 2: SEER
            '--observation_type',
            choices=[0, 1, 2],
            type=int,
            default=1
        )
        b_parser.add_argument(
            '-r',
            '--remote_address',
            help='Set this to an ipv4 address, when playing on a remote server or on a private subnet (with multiple '
                 'humans). For example -r 192.168.178.26 when you want to connect your friends machines in a private '
                 'subnet to the machine running the server at 192.168.178.26 or -r hanabi.live when you want to play on '
                 'the official server. Unfortunately, it currently does not work with eduroam.',
        )

    _add_args()
    arguments = argparser.parse_args()

    def _commands_valid(args):

        assert len(args.__dict__) > 1, 'Enter either client.py "agents_only" or client.py "with_human"'
        """ This function returns True, iff the user specified input does not break the rules of the game"""

        if args.subparser_name == 'with_human':
            # assert legal number of total players
            assert 1 < args.num_humans + len(args.agent_classes) < 6
        elif args.subparser_name == 'agents_only':
            # assert table name only contains characters, as otherwise the server will complain
            assert args.table_name.isalpha()
        # ... whatever else will come to my mind

        return True

    assert _commands_valid(arguments)

    return arguments


if __name__ == "__main__":

    clients = []
    client_threads = []

    try:
        args = init_args()
        print(f"---------------------------------------------------------------------------\n"
              f"  Client started with following args:                                      \n"
              f"---------------------------------------------------------------------------\n"
              f"{args}\n")
        # Returns subnet ipv4 in private network and localhost otherwise
        addr, referer, addr_ws = get_addrs(args)

        # Run each client in a seperate thread
        num_agents = len(args.agent_classes)
        for i in range(num_agents):
            # set username equal to agent_class + number of agent
            agent_name = AGENT_CLASSES[args.agent_classes[i]] + '0' + str(i)

            # Login to Zamiels server (session-based)
            session, cookie = login(url='http://' + addr, referer=referer, username=agent_name, num_client=i)

            # Connect the agent to websocket url
            upgrade_to_websocket(url=addr_ws, session=session, cookie=cookie)

            # Start client (agent + websocket + rl_env game wrapper)
            c = Client('ws://' + addr + '/ws', cookie, username=agent_name, cmd_args=args)
            clients.append(c)

            # append to lists, s.t. threads can be started later and their objects dont get removed from garbage collector
            client_thread = threading.Thread(target=c.run)
            client_threads.append(client_thread)
            print(f"-------------------------------------------------------------------------\n"
                  f" Joined {agent_name} :                                                    \n"
                  f"-------------------------------------------------------------------------\n")
            time.sleep(.5)

        for thread in client_threads:
            thread.start()

        while True:
            # needed to keep the main thread alive, because its the only one that can catch the KeyboardInterrupt Signal
            # see https://stackoverflow.com/questions/19652446/python-program-with-thread-cant-catch-ctrlc
            time.sleep(1)

    except KeyboardInterrupt:
        print('\n Terminating game...')
        try:
            for c in clients:
                c.quit_game()
            sys.exit(0)
        except Exception:
            print('\n Force websocket shutdown...')
    # todo send gameJoin(gameID, password) when seelf.config['table_pw] is not '' for when -r is specified
    # self.config['table_pw'] shall not be '' if -r is specified
