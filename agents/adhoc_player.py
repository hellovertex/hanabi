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
import rainbow_agent as rainbow
import functools
from third_party.dopamine import logger


class AdHocRLPlayer(object):

    def __init__(self, observation_size, num_players, history_size, max_moves, type):

        agent_config = {
            "observation_size": observation_size,
            "num_players": num_players,
            "history_size": history_size,
            "max_moves": max_moves,
            "type": type
        }

        self.observation_size = agent_config["observation_size"]
        self.players = agent_config["num_players"]
        self.history_size = agent_config["history_size"]
        self.vectorized_observation_shape = agent_config["observation_size"]
        self.num_actions = agent_config["max_moves"]

        self.base_dir = None

        if agent_config["type"] == "10kit":
            self.base_dir = "/home/cawa/" # dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"
        elif agent_config["type"] == "20kit":
            self.base_dir = "/home/cawa/" # dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_20kit"
        elif agent_config["type"] == "custom_dis_punish":
            self.base_dir = "/home/cawa/" # dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_custom_r_discard_playable"
        else:
            print("AGENT TYPE UNKNOWN")
            sys.exit(0)

        ## Hard coded for now
        self.experiment_logger = logger.Logger('/home/cawa/" # dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/adhoc/rainbow_10kit_2_x_rainbow_20kit_2/logs')

        path_weights = os.path.join(self.base_dir,'checkpoints')

        self.agent = xp.create_agent(self.observation_size, self.num_actions, self.players, "Rainbow")

        start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_weights,"ckpt")
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

        return action
