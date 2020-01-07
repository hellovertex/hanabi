import numpy as np
from absl import flags

# from third_party.dopamine import checkpointer
# from third_party.dopamine import iteration_statistics
# from third_party.dopamine.discrete_domains import run_experiment
#
#
# from dopamine.agents.dqn import dqn_agent
# from dopamine.discrete_domains import run_experiment
# from absl import flags
#
#
# import dqn_agent
# import gin.tf
# import rl_env
# import numpy as np
# import rainbow_agent
# import tensorflow as tf

import collections
import os
import sys
sys.path.append(os.path.abspath('../hanabi_learning_environment/agents/rainbow/'))

# from dopamine.colab import utils as colab_utils
from third_party.dopamine import utils

import seaborn as sns
import matplotlib.pyplot as plt


# BASE_PATH = '/tmp/colab_dope_run'  # @param
GAMES = ['Hanabi-Small-Seer']  # @param

parameter_set = collections.OrderedDict([
    ('agent', ['000dqn','010dqn','001dqn','011dqn']),
    ('game', GAMES),
])


sample_data = utils.read_experiment(
    os.path.abspath('../training'),
    parameter_set=parameter_set,
    job_descriptor='{}',
    #summary_keys=['train_episode_returns', 'train_episode_lengths', 'eval_episode_returns', 'eval_episode_lengths'])
    summary_keys = ['train_episode_returns', 'eval_episode_returns'])

#sample_data['agent'] = 'Tiny DQN'
sample_data['run_number'] = 1
sample_data['eval_episode_returns'][sample_data['eval_episode_returns']==-1] = np.NaN #replace -1 with NaN for nicer plotting
#sample_data['eval_episode_lengths'][sample_data['eval_episode_lengths']==-1] = np.NaN

fig,ax = plt.subplots()
sns.tsplot(data=sample_data, time='iteration', unit='run_number',
         condition='agent', value='train_episode_returns', ax=ax)
plt.scatter(sample_data.iteration, sample_data.eval_episode_returns, color='red')
plt.title("returns over episodes")
plt.show()

# fig,ax = plt.subplots()
# sns.tsplot(data=sample_data, time='iteration', unit='run_number',
#          condition='agent', value='train_episode_lengths', ax=ax)
# plt.scatter(sample_data.iteration, sample_data.eval_episode_lengths, color='red')
# plt.title("game lenght over episodes")
#
# plt.show()



#
# sample_data = colab_utils.read_experiment(
#     os.path.abspath('./dqn_mini'),
#     parameter_set=parameter_set,
#     job_descriptor='{}')
#
# sample_data['agent'] = 'Mini DQN'
# sample_data['run_number'] = 1
#
# # for game in GAMES:
# #   experimental_data[game] = experimental_data[game].merge(
# #       sample_data[sample_data.game == game], how='outer')
#
# @title Plot the sample agent data against the baselines.