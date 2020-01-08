from tf_agents_lib.wrappers import PyEnvironmentBaseWrapper
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec
import numpy as np

discount = np.asarray(.999, dtype=np.float32)
dtype_vectorized = 'float'


class PubMDPWrapper(PyEnvironmentBaseWrapper):
    """Wrapper derived from PyEnvironmentBaseWrapper(py_environment.PyEnvironment)
    for the PubMDP sitting on the HLE
    """

    def __init__(self, env):
        super(PyEnvironmentBaseWrapper, self).__init__()
        self.pub_mdp = env
        self._episode_ended = False

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.pub_mdp, name)

    @property
    def batched(self):
        return getattr(self.pub_mdp, 'batched', False)

    @property
    def batch_size(self):
        return getattr(self.pub_mdp, 'batch_size', None)

    def get_mask_legal_moves(self, observation):
        """ Given an observation,
        returns a boolean mask indicating whether actions are legal or not """

        legal_moves_as_int = observation['legal_moves_as_int']
        mask = np.full(self.pub_mdp.env.num_moves(), np.finfo(np.float32).min)

        mask[legal_moves_as_int] = 0

        return mask.astype(np.float32)

    def _reset(self, rewards_config=None):
        """Must return a tf_agents.trajectories.time_step.TimeStep namedTubple obj"""
        # i.e. ['step_type', 'reward', 'discount', 'observation']
        self._episode_ended = False
        observations = self.pub_mdp.reset()
        observation = observations['player_observations'][observations['current_player']]
        # reward is 0 on reset
        reward = np.asarray(0, dtype=np.float32)

        # oberservation is currently a dict, extract the 'vectorized' object
        obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)
        mask_valid_actions = self.get_mask_legal_moves(observation)
        score = np.array(self.pub_mdp.env.state.score())
        custom_rewards = {'hint_reward': 0,
                          'play_reward': 0,
                          'discard_reward': 0}

        obs = {'state': obs_vec, 'mask': mask_valid_actions, 'score': score, 'custom_rewards': custom_rewards}

        timestep = TimeStep(StepType.FIRST, reward, discount, obs)

        return timestep

    def _step(self, action):
        """Must return a tf_agents.trajectories.time_step.TimeStep namedTuple obj"""

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        action = int(action)
        observations, reward, done, custom_rewards = self.pub_mdp.step(action)
        observation = observations['player_observations'][observations['current_player']]
        # print(f'observation inside step function = {observation}')

        reward = np.asarray(reward, dtype=np.float32)

        obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)
        mask_valid_actions = self.get_mask_legal_moves(observation)
        score = np.array(self.pub_mdp.env.state.score())
        obs = {'state': obs_vec, 'mask': mask_valid_actions, 'score': score, 'custom_rewards': custom_rewards}

        if done:
            self._episode_ended = True
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        timestep = TimeStep(step_type, reward, discount, obs)
        self.pub_mdp.last_time_step = timestep
        # print(f'timestep from stepping = {timestep}')
        return timestep

    def observation_spec(self):
        """Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s."""

        # i.e. ('_shape', '_dtype', '_name', '_minimum', '_maximum')

        state_spec = BoundedArraySpec(
            shape=self.pub_mdp.vectorized_observation_shape(),
            dtype=dtype_vectorized,
            minimum=0,
            maximum=3,
            name='state'
        )
        mask_spec = ArraySpec(
            shape=(self.pub_mdp.env.num_moves(), ),
            dtype=np.float32,
            name='mask')

        return {'state': state_spec, 'mask': mask_spec}

    def action_spec(self):
        """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
        # inside self._env.step, action is converted to an int in, range [0, self._env.num_moves()]
        shape = ()
        dtype = np.int64
        minimum = 0
        maximum = self.pub_mdp.env.num_moves() - 1
        return BoundedArraySpec(shape, dtype, minimum, maximum, name='action')
