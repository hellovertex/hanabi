from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys
import numpy as np
# Rainbow Agent imports from hle
from agents.rainbow_copy.third_party.dopamine import checkpointer, logger
import agents.rainbow_copy.rainbow_agent as rainbow

# PPOAgent imports
from tf_agents.policies import py_tf_policy
from hanabi_learning_environment import rl_env
from training.tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper
from tf_agents.environments import tf_py_environment
from training.tf_agents_lib.masked_networks import MaskedActorDistributionNetwork
from training.tf_agents_lib.masked_networks import MaskedValueEvalNetwork
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.trajectories.time_step import TimeStep

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
    'ppo': 'PPOGuiAgent'
}

# todo make checkpoint paths configurable
ppo_ckpt_dir = '/home/hellovertex/Documents/github.com/hellovertex/hanabi/training/summaries/this_is_where_the_current_tfevent_files_will_get_stored/'
rainbow_ckpt_dir = '/agents/rainbow/'


class GUIAgent(object):
    """
    Abstract interface for GUI agents
    Subclass instantiation must make sure, that the trained agents loaded,
    are the ones that were trained for the very game variant that is played.
    """

    def __new__(cls, *args, **kwargs):
        assert cls.__name__ in AGENT_CLASSES.values()
        return super(GUIAgent, cls).__new__(cls)

    def act(self, observation_dict):
        """ Expects pyhanabi observation dict
        Returns action as dict or int"""
        raise NotImplementedError


class RainbowAgent(GUIAgent):

    def __init__(self, agent_config):

        tf.reset_default_graph()
        project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        # todo make this configurable
        self.base_dir = project_path + '/agents/rainbow_10kit/'

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

    def act(self, observation_dict):
        # Returns Integer Action
        action_int = self.agent._select_action(observation_dict["vectorized"], observation_dict["legal_moves_as_int_formated"])

        # Decode it back to dictionary object
        action_dict = observation_dict["legal_moves"][np.where(np.equal(action_int, observation_dict["legal_moves_as_int"]))[0][0]]

        return action_dict


class PPOGuiAgent(GUIAgent):

    def __init__(self, config, ckpt_dir=ppo_ckpt_dir):
        # --- Tf session --- #
        tf.reset_default_graph()
        self.sess = tf.Session()
        tf.compat.v1.enable_resource_variables()

        # --- Environment Stub--- #
        env = rl_env.HanabiEnv(config=config['game_config'])
        wrapped_env = PyhanabiEnvWrapper(env)
        tf_env = tf_py_environment.TFPyEnvironment(wrapped_env)
        time_step_spec = tf_env.time_step_spec()
        observation_spec = tf_env.observation_spec()
        action_spec = tf_env.action_spec()
        del env, wrapped_env, tf_env

        with self.sess.as_default():
            # --- Init Networks --- #
            actor_net = MaskedActorDistributionNetwork(  # set up actor network as trained before
                observation_spec(),
                action_spec(),
                fc_layer_params=(150, 75)
            )
            value_net = MaskedValueEvalNetwork(  # set up value network as trained before
                observation_spec(), fc_layer_params=(150, 75)
            )

            # --- Init agent --- #
            agent = PPOAgent(  # set up ppo agent with tf_agents
                time_step_spec(),
                action_spec(),
                actor_net=actor_net,
                value_net=value_net,
                train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
                normalize_observations=False
            )

            # --- init policy --- #
            self.policy = py_tf_policy.PyTFPolicy(agent.policy)
            self.policy.initialize(None)
            # --- restore from checkpoint --- #
            self.policy.restore(policy_dir=ckpt_dir, assert_consumed=False)

            # Run tf graph
            self.sess.run(agent.initialize())

    def act(self, observation_dict):
        NULL = 1

        # create tf_agents Timestep for tf_agents policy
        obs = {'state': observation_dict['vectorized'], 'mask': observation_dict['legal_moves_as_int_formated']}
        observation = TimeStep(step_type=NULL, reward=NULL, discount=NULL, observation=obs)
        with self.sess.as_default():
            # compute tf_agents PolicyStep
            policy_step = self.policy.action(observation)
            # convert integer action back to action dictioary
            action_int = policy_step.action
            action_dict = observation_dict["legal_moves"][np.where(np.equal(action_int, observation_dict["legal_moves_as_int"]))[0][0]]
            return action_dict
