from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec

import numpy as np
discount = np.asarray(.999, dtype=np.float32)  # todo get from somewhere else, probably gin?
dtype_vectorized = 'uint8'


class PyhanabiEnvWrapper(PyEnvironmentBaseWrapper):

  """Pyhanabi wrapper"""

  def __init__(self, env):
    super(PyEnvironmentBaseWrapper, self).__init__()
    self._env = env

  def __getattr__(self, name):
    """Forward all other calls to the base environment."""
    return getattr(self._env, name)

  @property
  def batched(self):
    return getattr(self._env, 'batched', False)

  @property
  def batch_size(self):
    return getattr(self._env, 'batch_size', None)

  def _reset(self):
    """Must return a tf_agents.trajectories.time_step.TimeStep namedTubple obj"""
    # i.e. ['step_type', 'reward', 'discount', 'observation']
    observations = self._env.reset()
    observation = observations['player_observations'][observations['current_player']]
    # reward is 0 on reset
    print("LEGAL_MOVES", observation['legal_moves'])
    reward = np.asarray(0, dtype=np.float32)
    observation = np.asarray((observation['vectorized'], observation['legal_moves_as_int']), dtype='O')
    print("AS INT", observation)

    return TimeStep(StepType.FIRST, reward, discount, observation)

  def _step(self, action):
    """Must return a tf_agents.trajectories.time_step.TimeStep namedTubple obj"""
    print("ACTION ", action)
    print("CORRESPONDING MOVE ", self._env.game.get_move(action))

    if isinstance(action, np.ndarray):
      action = int(action)
    observations, reward, done, info = self._env.step(action)

    print("------------------------------")
    print("Step successful")
    reward = np.asarray(reward, dtype=np.float32)

    observation = observations['player_observations'][observations['current_player']]
    observation = np.asarray((observation['vectorized'], observation['legal_moves_as_int']), dtype='O')

    if done:
        step_type = StepType.LAST
    else:
        step_type = StepType.MID

    return TimeStep(step_type, reward, discount, observation)

  def observation_spec(self):
    """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
    # i.e. ('_shape', '_dtype', '_name', '_minimum', '_maximum')

    shape = (2, )
    dtype = np.ndarray

    return ArraySpec(shape, dtype, name='observation')

  def action_spec(self):
    """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
    # inside self._env.step, action is converted to an int in range [0, self._env.num_moves()]
    shape = ()
    dtype = 'int64'
    minimum = 0
    maximum = self._env.num_moves() - 1
    return BoundedArraySpec(shape, dtype, minimum, maximum, name='action')
