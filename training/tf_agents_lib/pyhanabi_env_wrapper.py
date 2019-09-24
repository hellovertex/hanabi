from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec

import numpy as np

discount = np.asarray(.999, dtype=np.float32)  # todo get from somewhere else, probably gin?
dtype_vectorized = 'uint8'


class PyhanabiEnvWrapper(PyEnvironmentBaseWrapper):
    """Pyhanabi wrapper derived from PyEnvironmentBaseWrapper(py_environment.PyEnvironment) """

    def __init__(self, env):
        super(PyEnvironmentBaseWrapper, self).__init__()
        self._env = env
        self._episode_ended = False

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
        mask = np.full(self._env.num_moves(), np.finfo(np.float32).min)
        mask[legal_moves_as_int] = 0

        return mask.astype(np.float32)

    def _reset(self):
        """Must return a tf_agents.trajectories.time_step.TimeStep namedTubple obj"""
        # i.e. ['step_type', 'reward', 'discount', 'observation']
        self._episode_ended = False
        observations = self._env.reset()
        observation = observations['player_observations'][observations['current_player']]

        # reward is 0 on reset
        reward = np.asarray(0, dtype=np.float32)

        # oberservation is currently a dict, extract the 'vectorized' object
        obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)
        mask_valid_actions = self.get_mask_legal_moves(observation)
        obs = {'state': obs_vec, 'mask': mask_valid_actions}
        # (48, ) int64
        #print(mask_valid_actions.shape, mask_valid_actions.dtype)
        return TimeStep(StepType.FIRST, reward, discount, obs)

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
        observations, reward, done, info = self._env.step(action)
        observation = observations['player_observations'][observations['current_player']]

        reward = np.asarray(reward, dtype=np.float32)

        obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)
        mask_valid_actions = self.get_mask_legal_moves(observation)
        obs = {'state': obs_vec, 'mask': mask_valid_actions}

        if done:
            self._episode_ended = True
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(step_type, reward, discount, obs)

    def observation_spec(self):
        """Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s."""

        # i.e. ('_shape', '_dtype', '_name', '_minimum', '_maximum')

        state_spec = BoundedArraySpec(
            shape=self._env.vectorized_observation_shape(),
            dtype=dtype_vectorized,
            minimum=0,
            maximum=1,
            name='state'
        )
        mask_spec = ArraySpec(
            shape=(self._env.num_moves(), ),
            dtype=np.float32,
            name='mask')

        return {'state': state_spec, 'mask': mask_spec}

    def action_spec(self):
        """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
        # inside self._env.step, action is converted to an int in, range [0, self._env.num_moves()]
        shape = ()
        dtype = np.int64
        minimum = 0
        maximum = self._env.num_moves() - 1
        return BoundedArraySpec(shape, dtype, minimum, maximum, name='action')
