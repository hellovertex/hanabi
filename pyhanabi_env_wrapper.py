from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.specs.array_spec import BoundedArraySpec

import numpy as np
discount = .999  # todo get from somewhere else, probably gin?
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
    observation, reward, done, info = self._env.reset()
    step_type = StepType.FIRST

    # oberservation is currently a dict, extract the 'vectorized' object
    obs_vec = np.array(observation['vectorized'], dtype=dtype_vectorized)

    return TimeStep(step_type, reward, discount, obs_vec)

  def _step(self, action):
    """Must return a tf_agents.trajectories.time_step.TimeStep namedTubple obj"""
    return self._env.step(action)

  def observation_spec(self):
    """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
    # i.e. ('_shape', '_dtype', '_name', '_minimum', '_maximum')
    shape = self._env.vectorized_observation_shape()
    dtype = dtype_vectorized
    minimum = 0
    maximum = 1
    return BoundedArraySpec(shape, dtype, minimum, maximum)

  def action_spec(self):
    """Must return a tf_agents.specs.array_spec.BoundedArraySpec obj"""
    # inside self._env.step, action is converted to an int in range [0, self._env.num_moves()]
    shape = ()
    dtype = 'int64'
    minimum = 0
    maximum = self._env.num_moves()

    return BoundedArraySpec(shape, dtype, minimum, maximum)

  def render(self, mode='rgb_array'):
    return self._env.render(mode)

  def wrapped_env(self):
    # not needed as of now
    return self._env
