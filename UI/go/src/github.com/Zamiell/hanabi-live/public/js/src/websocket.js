/*
    Communication with the server is done through the WebSocket protocol
    The client uses a slightly modified version of the Golem WebSocket library
*/

// Imports
const chat = require('./chat');
const game = require('./game/main');
const golem = require('../lib/golem');
const globals = require('./globals');
const lobby = require('./lobby/main');
const modals = require('./modals');
const settings = require('./lobby/settings');
console.log("CALLED WEBSOCKET.JS")
exports.set = () => {
    // Connect to the WebSocket server
    let websocketURL = 'ws';
    if (window.location.protocol === 'https:') {
        websocketURL += 's';
    }
    websocketURL += '://';
    websocketURL += window.location.hostname;
    if (window.location.port !== '') {
        websocketURL += ':';
        websocketURL += window.location.port;
    }
    websocketURL += '/ws';
    console.log('Connecting to websocket URL:', websocketURL);
    const debug = window.location.pathname.includes('/dev');
    globals.conn = new golem.Connection(websocketURL, debug);
    // This will automatically use the cookie that we received earlier from the POST
    // If the second argument is true, debugging is turned on

    // Define event handlers
    globals.conn.on('open', () => {
        // We will show the lobby upon recieving the "hello" command from the server
        console.log('WebSocket connection established.');
    });
    globals.conn.on('close', () => {
        console.log('WebSocket connection disconnected / closed.');
        modals.errorShow('Disconnected from the server. Either your Internet hiccuped or the server restarted.');
    });
    globals.conn.on('socketError', (event) => {
        // "socketError" is defined in "golem.js" as mapping to the WebSocket "onerror" event
        console.error('WebSocket error:', event);

        if ($('#loginbox').is(':visible')) {
            lobby.login.formError('Failed to connect to the WebSocket server. The server might be down!');
        }
    });

    // All of the normal commands/messages that we expect from the server are defined in the
    // "initCommands()" function
    initCommands();

    globals.conn.send = (command, data) => {
        if (typeof data === 'undefined') {
            data = {};
        }
        if (window.location.pathname.includes('/dev')) {
            console.log(`%cSent ${command}:`, 'color: green;');
            console.log(data);
        }
        globals.conn.emit(command, data);
    };

    // Send any client errors to the server for tracking purposes
    window.onerror = (message, url, lineno, colno) => {
        // We don't want to report errors if someone is doing local development
        if (window.location.hostname === 'localhost') {
            return;
        }

        try {
            globals.conn.emit('clientError', {
                message,
                url,
                lineno,
                colno,
            });
        } catch (err) {
            console.error('Failed to transmit the error to the server:', err);
        }
    };
};

// This is all of the normal commands/messages that we expect to receive from the server
const initCommands = () => {
    globals.conn.on('hello', (data) => {
        // Store variables relating to our user account on the server
        globals.username = data.username; // We might have logged-in with a different stylization
        globals.totalGames = data.totalGames;
        globals.settings = data.settings;

        // Some settings are stored on the server as numbers,
        // but we need them as strings because they will exist in an input field
        const valuesToConvertToStrings = [
            'createTableBaseTimeMinutes',
            'createTableTimePerTurnSeconds',
        ];
        for (const value of valuesToConvertToStrings) {
            globals.settings[value] = globals.settings[value].toString();
        }

        $('#nav-buttons-history-total-games').html(globals.totalGames);
        settings.init();
        lobby.login.hide(data.firstTimeUser);

        if (!data.firstTimeUser) {
            // Validate that we are on the latest JavaScript code
            if (
                data.version !== globals.version
                // If the server is gracefully shutting down, then ignore the version check because
                // the new client code is probably not compiled yet
                && !data.shuttingDown
                && !window.location.pathname.includes('/dev')
            ) {
                let msg = 'You are running an outdated version of the Hanabi client code. ';
                msg += `(You are on <i>v${globals.version}</i> and the latest is <i>v${data.version}</i>.)<br />`;
                msg += 'Please perform a hard-refresh to get the latest version.<br />';
                msg += '(On Windows, the hotkey for this is "Ctrl + F5". ';
                msg += 'On MacOS, the hotkey for this is "Command + Shift + R".)';
                modals.warningShow(msg);
                return;
            }

            // Automatically go into a replay if surfing to "/replay/123"
            let gameID = null;
            const match = window.location.pathname.match(/\/replay\/(\d+)$/);
            if (match) {
                [, gameID] = match;
            } else if (window.location.pathname === '/dev2') {
                gameID = '51'; // The first game in the Hanabi Live database
            }
            if (gameID !== null) {
                setTimeout(() => {
                    gameID = parseInt(gameID, 10); // The server expects this as an integer
                    globals.conn.send('replayCreate', {
                        gameID,
                        source: 'id',
                        visibility: 'solo',
                    });
                }, 10);
            }
        }
    });

    globals.conn.on('user', (data) => {
        globals.userList[data.id] = data;
        lobby.users.draw();
    });

    globals.conn.on('userLeft', (data) => {
        delete globals.userList[data.id];
        lobby.users.draw();
    });

    globals.conn.on('table', (data) => {
        // The baseTime and timePerTurn come in seconds, so convert them to milliseconds
        data.baseTime *= 1000;
        data.timePerTurn *= 1000;

        globals.tableList[data.id] = data;
        lobby.tables.draw();
    });

    globals.conn.on('tableGone', (data) => {
        delete globals.tableList[data.id];
        lobby.tables.draw();
    });

    globals.conn.on('chat', (data) => {
        chat.add(data, false); // The second argument is "fast"
        if (
            data.room === 'game'
            && globals.ui !== null
            && !$('#game-chat-modal').is(':visible')
        ) {
            if (globals.ui.globals.spectating && !globals.ui.globals.sharedReplay) {
                // Pop up the chat window every time for spectators
                game.chat.toggle();
            } else {
                // Do not pop up the chat window by default;
                // instead, change the "Chat" button to say "Chat (1)"
                globals.chatUnread += 1;
                globals.ui.updateChatLabel();
            }
        }
    });

    // The "chatList" command is sent upon initial connection
    // to give the client a list of past lobby chat messages
    // It is also sent upon connecting to a game to give a list of past in-game chat messages
    globals.conn.on('chatList', (data) => {
        for (const line of data.list) {
            chat.add(line, true); // The second argument is "fast"
        }
        if (
            // If the UI is open, we assume that this is a list of in-game chat messages
            globals.ui !== null
            && !$('#game-chat-modal').is(':visible')
        ) {
            globals.chatUnread += data.unread;
            globals.ui.updateChatLabel();
        }
    });

    globals.conn.on('joined', () => {
        // We joined a new game, so transition between screens
        lobby.tables.draw();
        lobby.pregame.show();
    });

    globals.conn.on('left', () => {
        // We left a table, so transition between screens
        lobby.tables.draw();
        lobby.pregame.hide();
    });

    globals.conn.on('game', (data) => {
        globals.game = data;

        // The baseTime and timePerTurn come in seconds, so convert them to milliseconds
        globals.game.baseTime *= 1000;
        globals.game.timePerTurn *= 1000;

        lobby.pregame.draw();
    });

    globals.conn.on('tableReady', (data) => {
        if (data.ready) {
            $('#nav-buttons-pregame-start').removeClass('disabled');
        } else {
            $('#nav-buttons-pregame-start').addClass('disabled');
        }
    });

    globals.conn.on('gameStart', (data) => {
        if (!data.replay) {
            lobby.pregame.hide();
        }
        game.show(data.replay);
    });

    globals.conn.on('gameHistory', (dataArray) => {
        // data will be an array of all of the games that we have previously played
        for (const data of dataArray) {
            globals.historyList[data.id] = data;

            if (data.incrementNumGames) {
                globals.totalGames += 1;
            }
        }

        // The server sent us more games because
        // we clicked on the "Show More History" button
        if (globals.historyClicked) {
            globals.historyClicked = false;
            lobby.history.draw();
        }

        const shownGames = Object.keys(globals.historyList).length;
        $('#nav-buttons-history-shown-games').html(shownGames);
        $('#nav-buttons-history-total-games').html(globals.totalGames);
        if (shownGames === globals.totalGames) {
            $('#lobby-history-show-more').hide();
        }
    });

    globals.conn.on('historyDetail', (data) => {
        globals.historyDetailList.push(data);
        lobby.history.drawDetails();
    });

    globals.conn.on('sound', (data) => {
        if (globals.settings.sendTurnSound && globals.currentScreen === 'game') {
            if (globals.ui.globals.surprise) {
                globals.ui.globals.surprise = false;
                if (data.file === 'turn_other') {
                    data.file = 'turn_surprise';
                }
            }
            game.sounds.play(data.file);
        }
    });

    globals.conn.on('name', (data) => {
        globals.randomName = data.name;
    });

    globals.conn.on('warning', (data) => {
        console.warn(data.warning);
        modals.warningShow(data.warning);
        if (
            globals.currentScreen === 'game'
            && globals.ui !== null
            && globals.ui.globals.ourTurn
        ) {
            globals.ui.reshowClueUIAfterWarning();
        }
    });

    globals.conn.on('error', (data) => {
        console.error(data.error);
        modals.errorShow(data.error);

        // Disconnect from the server, if connected
        if (!globals.conn) {
            globals.conn.close();
        }
    });

    // There are yet more command handlers for events that happen in-game
    // These will only have an effect if the current screen is equal to "game"
    game.websocket.init();
};
