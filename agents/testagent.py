"""Playable class used to play games with the server"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

sys.path.insert(0, '/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/agent_classes')

import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

import agents.rainbow.dqn_agent as dqn
import agents.rainbow.run_experiment as xp
import vectorizer
import agents.rainbow.rainbow_agent as rainbow
import functools
from agents.rainbow.third_party.dopamine import logger

class RLPlayer(object):

    def __init__(self,agent,env,observation_size,history_size,tf_device='/cpu:*'):

        """Initializes the agent and constructs its graph.
        Vars:
          observation_size: int, size of observation vector on one time step.
          history_size: int, number of time steps to stack.
          graph_template: function for building the neural network graph.
          tf_device: str, Tensorflow device on which to run computations.
        """

        if env==None:
            print("Specify environment")
            return
        # We use the environment as a library to transform observations and actions(e.g. vectorize)
        self.env = env

        self.obs_stacker = xp.create_obs_stacker(self.env,history_size)

        self.num_actions = self.env.num_moves()

        self.observation_size = observation_size

        self.history_size = history_size

        if agent=="DQN":

            self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/experiments/full_4pl_2000it/"

            self.experiment_logger = logger.Logger('{}/logs'.format(self.base_dir))

            path_dqn = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/experiments/full_4pl_2000it/checkpoints"
            self.agent = xp.create_agent(self.env,self.obs_stacker,"DQN")

            start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_dqn,"ckpt")

            print("\n---------------------------------------------------")
            print("Creating agent from trained model at iteration: {}".format(start_iteration))
            print("---------------------------------------------------\n")

            self.agent.eval_mode = True

        elif agent == "Rainbow":

            self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_test"

            self.experiment_logger = logger.Logger('{}/logs'.format(self.base_dir))

            path_rainbow = os.path.join(self.base_dir,'checkpoints')
            # path_rainbow = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit/checkpoints"
            self.agent = xp.create_agent(self.env,self.obs_stacker,"Rainbow")
            self.agent.eval_mode = True

            start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_rainbow,"ckpt")
            print("\n---------------------------------------------------")
            print("Creating agent from trained model at iteration: {}".format(start_iteration))
            print("---------------------------------------------------\n")

        else:
            print("Specify Agent")
            return

    def load_model_weights(self,path,iteration_number):

        self.saver = tf.train.Saver()
        self.saver.restore(self._sess,
                            os.path.join(path,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True

    '''
    args:
        observation: expects an already vectorized observation from vectorizer.ObservationVectorizer
    returns:
        an integer, representing the appropriate action to take
    '''

    def act(self, observation):

        # Returns Integer Action
        action = self.agent._select_action(observation["vectorized"], observation["legal_moves_as_int_formated"])

        # Decode it back to dictionary object
        move_dict = observation["legal_moves"][np.where(np.equal(action,observation["legal_moves_as_int"]))[0][0]]

        return move_dict
