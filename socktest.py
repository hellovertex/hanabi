import requests
import websocket


def login(url, referer="http://192.168.178.26/"):
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

    return url, cookie


def run_websocket(url, cookie):

    def on_message(ws, message):
        print("message = ", message)

    def on_data(ws, data):
        print("data", data)
    ws = websocket.WebSocketApp(url=url,
                                on_message=on_message,
                                cookie=cookie)
    ws.run_forever()


addr = '192.168.178.26'
session, cookies = login(url='http://' + addr)
url_ws, cookie = upgrade_to_websocket(url='http://' + addr + '/ws', session=session, cookies=cookies)
run_websocket(url=url_ws, cookie=cookie)

