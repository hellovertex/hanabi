from pathlib import Path
path_simple_agent = Path(__file__ + 'agents' + 'simple_agent')
path_rainbow_agent = Path(__file__ + 'agents' + 'rainbow_agent')

""" Client looks up available agents here, for creating their class instance(s). 
Set(config['agent_classes']) must be a subset of agent_classes.keys() """
AGENT_CLASSES = {
    'simple': {
        'filepath': path_simple_agent,
        'class': 'SimpleAgent',  # client calls eval(AGENT_CLASSES['class'])(AGENT_CLASSES['agent_config'])
    },
    'rainbow': {
        'filepath': path_rainbow_agent,
        'class': 'RainbowAgent',  # client calls eval(AGENT_CLASSES['class'])(AGENT_CLASSES['agent_config'])
    },
    # You can add your own agents here, simply follow this structure:
    # 'name': {
    # 'filepath': path_sexy_agent,
    # 'class': 'MySexyAgent'
    # }
}
