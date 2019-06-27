"""Playable class used to play games with the server"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/rainbow/')
sys.path.append(rel_path)

import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

import dqn_agent as dqn
import run_experiment_ui as xp
import vectorizer
import rainbow_agent as rainbow
import functools
from third_party.dopamine import logger

class RLPlayer(object):

    def __init__(self,agent_config):

        """
        Main Interface that allows a trained agent to interact with other Hanabi-Environments
        """

        self.observation_size = agent_config["observation_size"]
        self.players = agent_config["players"]
        self.history_size = agent_config["history_size"]
        self.vectorized_observation_shape = agent_config["observation_size"]

        self.obs_stacker = xp.create_obs_stacker(self.history_size, self.vectorized_observation_shape, self.players)
        self.num_actions = agent_config["max_moves"]
        #self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_test"
        #self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"


        self.base_dir = "/home/cawa/Documents/SoSe19/NIP/hanabi/agents/rainbow_10kit/"
        # self.base_dir = "/home/cawa/Documents/SoSe19/NIP/hanabi/"



        #self.base_dir = "/home/grinwald/Projects/TUB/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"
        self.experiment_logger = logger.Logger('{}/logs'.format(self.base_dir))

        path_rainbow = os.path.join(self.base_dir,'checkpoints')
        #print(path_rainbow)

        self.agent = xp.create_agent(self.observation_size, self.num_actions, self.players, "Rainbow")
        # print("====================")
        # print("Created Agent successfully")
        # print("====================")
        self.agent.eval_mode = True
        self.agent.partial_reload = True
        # print(self.agent.partial_reload)

        start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_rainbow,"ckpt")
        print("\n---------------------------------------------------")
        print("Initialized Model weights at start iteration: {}".format(start_iteration))
        print("---------------------------------------------------\n")

    '''
    args:
        observation: expects an already vectorized observation from vectorizer.ObservationVectorizer
    returns:
        action dict object
    '''

    def act(self, observation):

        # Returns Integer Action
        action = self.agent._select_action(observation["vectorized"], observation["legal_moves_as_int_formated"])
        # print(action)

        # Decode it back to dictionary object
        move_dict = observation["legal_moves"][np.where(np.equal(action,observation["legal_moves_as_int"]))[0][0]]

        return move_dict
