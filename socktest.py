#!/usr/bin/python3
""" PYTHON IMPORTS """
import requests
import websocket
import threading
import time
# try:
#     import thread
# except ImportError:
#     import _thread as thread

""" PROJECT LVL IMPORTS """
import commandsWebSocket as cmd


gameStarted = False

def login(url, referer):
    """ Make POST request to /login page"""
    ses = requests.session()
    # set up header
    # set useragent to chrome, s.t. we dont have to deal with the additional firefox warning
    ses.headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) snap Chromium/74.0.3729.131 Chrome/74.0.3729.131 Safari/537.36'
    ses.headers['Accept'] = '*/*'
    ses.headers['Accept-Encoding'] = 'gzip, deflate'  # only shows on firefox
    ses.headers['Accept-Language'] = 'en-US,en;q=0.5' # only shows on firefox
    ses.headers['Origin'] = url

    url = url+'/login'
    data = {
        "username": "big_python",  # try developer account
        "password": "01c77cf03d35866f8486452d09c067f538848058f12d8f005af1036740cccf98"
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

    # store cookies because we need them for web socket establishment

    cookieJar = ses.cookies
    print(response, referer)
    print(cookieJar.items())
    assert response.status_code == 200  # assert login was successful
    print(f"COOKIES: {cookieJar.items()[0]}")

    return ses, cookieJar


def upgrade_to_websocket(url, session, cookies):
    """ Make a GET request to UPGRADE HTTP to Websocket"""
    # cookies is a cookieJar obj which stores [(id, cookie)]
    id = cookies.items()[0][0]
    cookie = cookies.items()[0][1]

    # headers
    headers = {
        "Connection": "Upgrade",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Upgrade": "websocket",
        "Sec-Websocket-Version": "13",
        "Cookie": f"{id}={cookie}",
        "Sec-WebSocket-Key": "6R/+PZjR24OnAaKVwAnTRA==",  # Base64 encoded
        "Sec-Websocket-Extensions": "permessage-deflate; client_max_window_bits"
    }
    # make get
    response = session.get(url=url, headers=headers)
    print(f"STATUSCODE WS: {response.status_code}")

    assert response.status_code == 101  # 101 means SWITCHING_PROTOCOL, i.e. success

    return id+'='+cookie


def connect_websocket(url, cookie):
    print("Inside connect")
    print(cookie)
    # websocket.enableTrace(True)

    def on_message(ws, message):
        print("message = ", message)

    def on_error(ws, error):
        print("error")
        print(error)

    def on_close(ws):
        print("### closed ###")

    def on_data(ws, data):
        print("data", data)


    def on_open(ws):
        pass
        # thread.start_new_thread(ws.run_forever)
        # def run(*args):
        #     for i in range(3):
        #         time.sleep(1)
        #         ws.send("Hello %d" % i)
        #     time.sleep(1)
        #     ws.close()
        #     print("thread terminating...")
        #
        # thread.start_new_thread(run, ())

    ws = websocket.WebSocketApp(url=url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                cookie=cookie)
    ws.on_open = on_open
    thread.start_new_thread(ws.run_forever, ())

    join_game(ws, 1)


def join_game(ws, game_id):
    print("Joining game")
    #ws.send(f"gameJoin gameId: {1}")
    return


def get_local_ip():
    pass

if __name__ == "__main__":
    # addr = get_local_ip()  # TODO
    # addr = "localhost"  # TODO fix setting of cookies, s.t. we can run anywher
    addr = '192.168.178.26'
    referer = "http://192.168.178.26/"
    # referer = "http://localhost/"
    session, cookies = login(url='http://' + addr, referer=referer)
    cookie = upgrade_to_websocket(url='http://' + addr + '/ws', session=session, cookies=cookies)

    url='ws://' + addr + '/ws'
    # connect_websocket(url='ws://' + addr + '/ws', cookie=cookie)
    def on_message(ws, message):


        print("message = ", message)
        if "gameStarted" in message:
            ws.send(cmd.hello())
        if "init" in message:
            ws.send(cmd.ready())

    def on_error(ws, error):
        print("Error: ", error)

    def on_close(ws):
        print("### closed ###")

    def on_data(ws, data):
        print("data", data)


    def on_open(ws):
        pass

    ws = websocket.WebSocketApp(url=url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                cookie=cookie)
    ws.on_open = on_open
    # websocket.enableTrace(True)
    # ws.run_forever()
    # thread.start_new_thread(ws.run_forever, ())
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    gottaJoin = True

    print("GOTTA JOIN MAN")
    conn_timeout = 5
    while not ws.sock.connected and conn_timeout:
        time.sleep(1)
        conn_timeout -= 1

    while ws.sock.connected:
        if gottaJoin:
            throttle = 2
            time.sleep(throttle)
            ws.send(cmd.gameJoin(gameID='3'))
            ws.send(cmd.hello())
            gottaJoin = False
        # if gameStarted:
        #     ws.send(cmd.hello())
        #     ws.send(cmd.ready())
        time.sleep(0.1)
