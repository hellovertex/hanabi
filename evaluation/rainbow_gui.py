"""Playable class used to play games with the server"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import agents.rainbow_copy.run_experiment_ui as xp
import agents.rainbow_copy.rainbow_agent as rainbow
from agents.rainbow_copy.third_party.dopamine import logger
import os
import numpy as np

class RainbowPlayer(object):

    def __init__(self, agent_config):

        tf.reset_default_graph()
        project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        self.base_dir = project_path + '/evaluation/trained_models/'

        self.observation_size = agent_config["observation_size"]
        self.num_players = agent_config["num_players"]
        self.history_size = agent_config["history_size"]
        self.vectorized_observation_shape = agent_config["observation_size"]
        self.num_actions = agent_config["max_moves"]

        self.experiment_logger = logger.Logger(self.base_dir+'/logs')

        self.agent = rainbow.RainbowAgent(
            observation_size=self.observation_size,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_layers=1
            )

        path_weights = os.path.join(self.base_dir,'checkpoints')
        start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_weights,"ckpt")

        print("\n---------------------------------------------------")
        print("Initialized Model weights at start iteration: {}".format(start_iteration))
        print("---------------------------------------------------\n")

    def act(self, observation):
        # Returns Integer Action
        action_int = self.agent._select_action(observation["vectorized"], observation["legal_moves_as_int_formated"])

        # Decode it back to dictionary object
        action_dict = observation["legal_moves"][np.where(np.equal(action_int, observation["legal_moves_as_int"]))[0][0]]

        return action_dict

