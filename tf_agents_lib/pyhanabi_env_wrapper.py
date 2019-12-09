from .wrappers import PyEnvironmentBaseWrapper
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec

import numpy as np

dtype_vectorized = 'uint8'

class PyhanabiEnvWrapper(PyEnvironmentBaseWrapper):
    """Pyhanabi wrapper derived from PyEnvironmentBaseWrapper(py_environment.PyEnvironment) """

    def __init__(self, env, gamma = 0.99):
        super(PyEnvironmentBaseWrapper, self).__init__()
        self._env = env
        self._episode_ended = False
        self.gamma = np.asarray(0.99, dtype = 'float32')
        self._observation_spec = self.compute_observation_spec(self._env)
        
    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._env, name)

    @property
    def batched(self):
        return getattr(self._env, 'batched', False)

    @property
    def batch_size(self):
        return getattr(self._env, 'batch_size', None)

    def get_mask_legal_moves(self, observation):
        """ Given an observation,
        returns a boolean mask indicating whether actions are legal or not """

        legal_moves_as_int = observation['legal_moves_as_int']
        mask = np.full(self._env.num_moves(), -np.inf)
        mask[legal_moves_as_int] = 0

        return mask.astype(np.float32)

    def _reset(self, rewards_config = {}):
        """Must return a tf_agents.trajectories.time_step.TimeStep namedTubple obj"""
        # i.e. ['step_type', 'reward', 'discount', 'observation']
        self._episode_ended = False
        observations = self._env.reset(rewards_config)
        observation = observations['player_observations'][observations['current_player']]

        # reward is 0 on reset
        reward = np.asarray(0, dtype=np.float32)

        # oberservation is currently a dict, extract the 'vectorized' object
        obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)
        beliefs_prob_dict = observations['beliefs_prob_dict']
        mask_valid_actions = self.get_mask_legal_moves(observation)
        score = np.array(self._env.state.score())
        custom_rewards =  {'hint_reward' : 0,
                           'play_reward' : 0,
                           'discard_reward' : 0 }
        
        obs = {'state': obs_vec, 'mask': mask_valid_actions, 'score': score, 
               'custom_rewards' : custom_rewards, 'beliefs_prob_dict' : beliefs_prob_dict}

        return TimeStep(StepType.FIRST, reward, 1, obs)

    def _step(self, action):
        """Must return a tf_agents.trajectories.time_step.TimeStep namedTuple obj"""

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        #print('#### TYPE OF ACTION', type(action))
        #if isinstance(action, np.ndarray):
        action = int(action)
        #print('#### TYPE OF ACTION', type(action))
        observations, reward, done, custom_rewards = self._env.step(action)
        observation = observations['player_observations'][observations['current_player']]

        reward = np.asarray(reward, dtype=np.float32)

        obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)
        beliefs_prob_dict = observations['beliefs_prob_dict']
        mask_valid_actions = self.get_mask_legal_moves(observation)
        # stores current game score
        score = np.array(self._env.state.score())
        obs = {'state': obs_vec, 'mask': mask_valid_actions, 'score': score, 
               'custom_rewards' : custom_rewards, 'beliefs_prob_dict' : beliefs_prob_dict}

        if done:
            self._episode_ended = True
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(step_type, reward, 1, obs)


    @staticmethod
    def compute_observation_spec(environment):
        """
        Returns an `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        i.e. ('_shape', '_dtype', '_name', '_minimum', '_maximum')
        The shapes are dynamically computed, depending on whether the environment observation are augmented, or not.
        This is depending on the environment flag augment_input
        Args:
            environment: a rl_env.HanabiEnv object
        Returns: observation_spec dictionary with tf_agents.specs.array_spec.ArraySpec objects containg
        """
        # make sure we use the correct rl_env.py
        assert hasattr(environment, 'augment_input')
        maybe_additional_inputs = 0  # will be incremented in case of augmented observation
        maybe_additional_range = 0  # will be incremented in case of augmented observation

        if environment.augment_input:
            # observation is augmented, so adjsut the obs_spec accordingly
            if environment.augment_input_using_binary:
                maybe_additional_inputs += environment.game.num_players() * environment.game.hand_size() \
                                           * (environment.game.num_colors() + environment.game.num_ranks())
            else:
                maybe_additional_inputs += environment.game.num_players() * environment.game.hand_size()
                maybe_additional_range = environment.game.num_colors() + environment.game.num_ranks()


        # if we use OPEN_HANDS game mode
        assert hasattr(environment, 'open_hands')  # use correct version of env
        if environment.open_hands:
            # the cards are only revealed to acting player, hence the number of extra bits is
            # hand_size * num_colors * num_ranks
            num_extra_bits = environment.game.hand_size() * \
                             environment.game.num_colors() * environment.game.num_ranks()
            maybe_additional_inputs += num_extra_bits

        len_obs = environment.vectorized_observation_shape()[0]  # get length of vectorized observation

        state_spec = BoundedArraySpec(
            shape=(len_obs + maybe_additional_inputs, ),  # shape is expected to be tuple (N, )
            dtype=dtype_vectorized,
            minimum=0,
            maximum=1 + maybe_additional_range,
            name='state'
        )
        mask_spec = ArraySpec(
            shape=(environment.num_moves(),),
            dtype=np.float32,
            name='mask')
        info_spec = ArraySpec(
            shape=(),
            dtype=np.int,
            name='info'
        )
        
        return {'state': state_spec, 'mask': mask_spec, 'info': info_spec}

    def observation_spec(self):
        """Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s."""
        return self._observation_spec


    def action_spec(self):
        """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
        # inside self._env.step, action is converted to an int in, range [0, self._env.num_moves()]
        shape = ()
        dtype = np.int64
        minimum = 0
        maximum = self._env.num_moves() - 1
        return BoundedArraySpec(shape, dtype, minimum, maximum, name='action')
