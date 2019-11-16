from tf_agents.metrics import py_metrics, tf_py_metric
from tf_agents.utils import numpy_storage
from collections import namedtuple
import tensorflow as tf
import numpy as np

class PyScoreMetric(py_metrics.StreamingMetric):
    def __init__(self, name='PyScoreMetric', buffer_size=10, batch_size=None):
        super(PyScoreMetric, self).__init__(name=name, buffer_size=buffer_size, batch_size=batch_size)
        self._np_state = numpy_storage.NumpyState()
        self._np_state.episode_score = np.float64(0)

    def _reset(self, batch_size):
        self.score = 0
        self._np_state.episode_score = np.zeros(shape=(batch_size, ), dtype=np.float64)

    def _batched_call(self, trajectory):
        # we changed the environment wrapper, such that it contains 'info' field, holding the current game score
        episode_score = self._np_state.episode_score
        is_first = np.where(trajectory.is_first())
        episode_score[is_first] = 0

        episode_score = trajectory.observation['info']

        is_last = np.where(trajectory.is_last())
        self.add_to_buffer(episode_score[is_last])


class TfScoreMetric(tf_py_metric.TFPyMetric):
  """Metric to compute the average score."""

  def __init__(self, name='TfScoreMetric', dtype=tf.float32, buffer_size=10, batch_size=None):
    py_metric = PyScoreMetric(buffer_size=buffer_size, batch_size=batch_size)

    super(TfScoreMetric, self).__init__(
        py_metric=py_metric, name=name, dtype=dtype)
