from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys
import numpy as np
# Rainbow Agent imports from hle
from agents.rainbow_copy.third_party.dopamine import checkpointer, logger
import agents.rainbow_copy.rainbow_agent as rainbow


# make sure other project files find the config from wherever they are being called
path = os.path.dirname(sys.modules['__main__'].__file__)

# possible agent clients
AGENT_CLASSES = {
    """ 
    AGENT_CLASSES:
    
    provides the possible args for the clients agent type.
    E.g.
        python client.py -n 0 -a simple simple rainbow
    
    will join 3 AI agents to the gui server lobby (2 simple agents and 1 rainbow agent).
    Since -n=0, they do not expect human players and will start playing automatically, once their weights are fully loaded.
    
    You can add your own agents here
    just make sure they match the imports in client.py and if not, simply add a corresponding import statement there
    """
    'simple': 'SimpleAgent',  # as in deepminds hanabi-learning-environment
    'rainbow': 'RainbowAgent',  # as in deepminds hanabi-learning-environment
    'ppo': 'PPOAgent'
}


class GUIAgent(object):
    """
    Abstract interface for GUI agents
    Subclass instantiation must make sure, that the trained agents loaded,
    are the ones that were trained for the very game variant that is played.
    """

    def __new__(cls, *args, **kwargs):
        assert cls.__name__ in AGENT_CLASSES.values()
        return super(GUIAgent, cls).__new__(cls)

    def act(self, observation):
        """ Expects pyhanabi observation dict
        Returns action as dict or int"""
        raise NotImplementedError


class RainbowAgent(object):

    def __init__(self, agent_config):

        tf.reset_default_graph()
        project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        self.base_dir = project_path + '/evaluation/trained_models/'

        self.observation_size = agent_config["observation_size"]
        self.num_players = agent_config["num_players"]
        # self.history_size = agent_config["history_size"]
        # self.vectorized_observation_shape = agent_config["observation_size"]
        self.num_actions = agent_config["max_moves"]

        self.experiment_logger = logger.Logger(self.base_dir+'/logs')

        self.agent = rainbow.RainbowAgent(
            observation_size=self.observation_size,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_layers=1
            )

        checkpoint_dir = os.path.join(self.base_dir,'checkpoints')
        experiment_checkpointer = checkpointer.Checkpointer(checkpoint_dir, "ckpt")

        start_iteration = 0
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_dir)
        if latest_checkpoint_version >= 0:
            dqn_dictionary = experiment_checkpointer.load_checkpoint(latest_checkpoint_version)
            if self.agent.unbundle(checkpoint_dir, latest_checkpoint_version, dqn_dictionary):
                assert 'logs' in dqn_dictionary
                assert 'current_iteration' in dqn_dictionary

                self.experiment_logger.data = dqn_dictionary['logs']
                start_iteration = dqn_dictionary['current_iteration'] + 1
                tf.logging.info('Reloaded checkpoint and will start from iteration %d', start_iteration)

        print("\n---------------------------------------------------")
        print("Initialized Model weights at start iteration: {}".format(start_iteration))
        print("---------------------------------------------------\n")

    def act(self, observation):
        # Returns Integer Action
        action_int = self.agent._select_action(observation["vectorized"], observation["legal_moves_as_int_formated"])

        # Decode it back to dictionary object
        action_dict = observation["legal_moves"][np.where(np.equal(action_int, observation["legal_moves_as_int"]))[0][0]]

        return action_dict


class PPOAgent(GUIAgent):
    def __init__(self, train_dir):
        pass

    def act(self, observation):
        pass
