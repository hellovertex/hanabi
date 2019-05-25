from pathlib import Path

path_simple_agent = Path(__file__ + 'agents' + 'simple_agent')
path_rainbow_agent = Path(__file__ + 'agents' + 'rainbow_agent')

""" agent_classes provides the possible args for the clients agent instantiation. 
The keys will be read inside the client via sys.argv[] """
agent_classes = {
    'simple': {'filepath': path_simple_agent, 'class': 'SimpleAgent'},
    'rainbow': {'filepath': path_rainbow_agent, 'class': 'RainbowAgent'},
    # You can add your own agents here, simply follow this structure:
    # 'sys argument': {'filepath': path_sexy_agent, 'class': 'MySexyAgent'},
}
""" If you have terminated the client while a game was running, the default behaviour will make the agents return to 
the game when the client is restarted. However, this is not always desired for instance when the number of players 
change etc, so you can run the client with the --resetted flag to start a new game """
