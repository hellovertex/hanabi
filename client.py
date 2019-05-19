#!/usr/bin/python3
""" PROJECT LVL IMPORTS """
import commandsWebSocket as cmd
""" PYTHON IMPORTS """
import requests
import websocket
import threading
import time
import re
from game_state_wrapper import GameStateWrapper
from agents.simple_agent import SimpleAgent


class Client:
    def __init__(self, url, cookie, username, agent_config):
        """ Client wrapped around the agents, so they can play on Zamiels server
         https://github.com/Zamiell/hanabi-live. They compute actions offline
         and send back the corresponding json-encoded action on their turn."""

        # Opens a websocket on url:80
        self.ws = websocket.WebSocketApp(url=url,
                                         on_message=lambda ws, msg: self.on_message(ws, msg),
                                         on_error=lambda ws, msg: self.on_message(ws, msg),
                                         on_close=lambda ws, msg: self.on_message(ws, msg),
                                         cookie=cookie)

        # Set on_open seperately as it does crazy things otherwise #pythonWebsockets
        self.ws.on_open = lambda ws: self.on_open(ws)

        # listen for incoming messages
        self.daemon = threading.Thread(target=self.ws.run_forever)
        self.daemon.daemon = True
        self.daemon.start()  # [do this before self.run(), s.t. we can hand over the daemon to another Thread]

        # Store incoming server messages here, to get observations etc.
        self.msg_buf = list()  # maybe replace this with a game_state object

        # throttle to avoid race conditions
        self.throttle = 0.05  # 50 ms

        # Tell the Client, where in the process of joining/playing we are
        self.gottaJoinGame = False
        self.gameHasStarted = False

        # Will always be set to the game created last (on the server side ofc)
        self.gameID = None

        # Stores observations for each agent
        self.game = GameStateWrapper(agent_config['players'])

        # Hanabi playing agent
        self.agent = SimpleAgent(agent_config)
        self.agent_name = username

        # configuration required for agent initialization
        self.config = agent_config

    def on_message(self, ws, message):
        """ Forwards messages to game state wrapper and sets flags for self.run() """
        print("message = ", message)
        # TODO: write parse_msg() method and just dijkstra on CONSTs here
        # JOIN GAME
        if message.strip().startswith('table'):  # notification opened game
            self._set_auto_join_game(message)

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

    @staticmethod
    def on_error(ws, error):
        print("Error: ", error)

    @staticmethod
    def on_close(ws):
        print("### closed ###")

    @staticmethod
    def on_data(ws, data):
        # print("data", data)
        pass

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
                # JOIN GAME
                if self.gottaJoinGame and self.gameID:
                    time.sleep(self.throttle)
                    self.ws.send(cmd.gameJoin(gameID=self.gameID))
                    self.gottaJoinGame = False

                if self.gameHasStarted:
                    # ON AGENTS TURN [we set the corresponding flag in self.on_message()] #todo
                    agent_id = self.game.players.index(self.agent_name)
                    # Compute ACTION
                    if self.game.players.index(self.agent_name) == self.game.cur_player:

                        obs = self.game.get_observation(agent_id)
                        a = self.agent.act(obs)
                        # Send to server
                        self.ws.send(self.game.parse_action_to_msg(a))

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


def login(url, referer, username="big_python", password="01c77cf03d35866f8486452d09c067f538848058f12d8f005af1036740cccf98"):
    """ Make POST request to /login page"""
    ses = requests.session()
    # set up header
    # set useragent to chrome, s.t. we dont have to deal with the additional firefox warning
    ses.headers[
        'User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) snap Chromium/74.0.3729.131 Chrome/74.0.3729.131 Safari/537.36'
    ses.headers['Accept'] = '*/*'
    ses.headers['Accept-Encoding'] = 'gzip, deflate'  # only shows on firefox
    ses.headers['Accept-Language'] = 'en-US,en;q=0.5'  # only shows on firefox
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
        "Referer": referer,  # todo: dynamically get addr
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
    print(f"STATUSCODE WS: {response.status_code}")

    assert response.status_code == 101  # 101 means SWITCHING_PROTOCOL, i.e. success

    return


def get_addrs():
    """ We will use this in private networks, to avoid localhost-related bugs for now.
    However, we have to get the localhost-settings running at some point."""
    # addr = get_local_ip()  # TODO
    #addr = "localhost"
    #referer = "http://localhost/"
    addr = '192.168.178.26'
    referer = "http://192.168.178.26/"
    return addr, referer


if __name__ == "__main__":
    """ maybe add following args: 
    - use_localhost // use_remote
    - num of agents (client instances) 
    - [AGENTCLASSES] 
    - lobbyname (e.g. for when playing online), used in run() method of the client"""

    # Returns subnet ipv4 in private network and localhost otherwise
    addr, referer = get_addrs()

    # Login to Zamiels server (session-based)
    session, cookie = login(url='http://' + addr, referer=referer)
    upgrade_to_websocket(url='http://' + addr + '/ws', session=session, cookie=cookie)

    # Connect the agent to websocket url
    url = 'ws://' + addr + '/ws'
    username = "big_python"  # todo: get this from the loop that we have to do from argvs
    config = {'players': 2}
    app = Client(url, cookie, username, config)
    app.run()

