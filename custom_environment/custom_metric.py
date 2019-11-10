from tf_agents.metrics import py_metrics, tf_py_metric
from tf_agents.utils import numpy_storage
from collections import namedtuple
import tensorflow as tf
import numpy as np

""" 
score_metric = S
eval_metrics.append(score_metric)
summary_op = StreamingMetric.tf_summaries()


eval_metrics are instances of 
- StreamingMetric <- PyStepMetric <- PyMetric

- question is how does observing the driver update the metrics CORRECTLY
- the eval_metrics(traj) is called at EVERY STEP
- how does the eval_metric know, when to empty its buffer, i.e. when the game has ended
==> The metrics always store self._np_state.episode_return

- Understand the namedtuple trajectory


metric_utils.compute_summaries(eval_metrics, eval_py_env, eval_py_policy)
- resets each metric
- runs driver on eval_py_env, using eval_py_policy, with observers=eval_metrics
- returns [eval_metric.result() for eval_metric in eval_metrics]






"""
traj = namedtuple('Trajectory', ['obs', 'action', 'reward'])
transition = traj(obs=55, action=155, reward=255)
#print(transition.reward)


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
  """Metric to compute the average return."""

  def __init__(self, name='PyScoreMetric', dtype=tf.float32, buffer_size=10):
    py_metric = py_metrics.AverageReturnMetric(buffer_size=buffer_size)

    super(TfScoreMetric, self).__init__(
        py_metric=py_metric, name=name, dtype=dtype)

next_step_type = np.array([1, 1, 2, 1, 1, 2, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 0])
print(np.where(next_step_type == 0))