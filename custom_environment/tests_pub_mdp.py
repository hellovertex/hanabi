import numpy as np

from hanabi_learning_environment import rl_env
from agents.simple_agent import SimpleAgent
from custom_environment.pub_mdp import  PubMDP
import os
"""
 (1) Generate observations using rl_env_example.py Runner()
 (2) Init PubMDP on top of it
 (3) run augment_observation()
 (4) Check if candidates and hint_mask is being updated correctly
 (5) Check V1 and BB computations
 """


class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = flags['game_config']
    self.environment = rl_env.HanabiEnv(flags['game_config'])
    self.agent_class = SimpleAgent
    self.pub_mdp = PubMDP(flags['game_config'])

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(self.flags['num_episodes']):
      observations = self.environment.reset()
      agents = [self.agent_class(self.agent_config)
                for _ in range(self.flags['players'])]
      done = False
      episode_reward = 0

      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]

          action = agent.act(observation)
          if observation['current_player'] == agent_id:
            augmented_observation = self.pub_mdp.augment_observation(observation)
            # print(augmented_observation)
            assert action is not None
            current_player_action = action

          else:
            assert action is None
        # Make an environment step.
        #print('Agent: {} action: {}'.format(observation['current_player'],
        #                                    current_player_action))
        observations, reward, done, unused_info = self.environment.step(
            current_player_action)

        episode_reward += reward
      rewards.append(episode_reward)

      #print('Running episode: %d' % episode)
      #print('Max Reward: %.3f' % max(rewards))
    return rewards


class ABCTest(object):
    def __init__(self, config=None):
        self.config = config
        if config is None:
            self.config = {  # config for Hanabi-Small
                "colors": 2,
                "ranks": 5,
                "players": 2,
                "hand_size": 2,
                "max_information_tokens": 3,
                "max_life_tokens": 1,
                "observation_type": 1}
        self.flags = {'num_episodes': 1, 'game_config': self.config, 'game_type': 'Hanabi-Full', 'players': 2}
        self.runner = Runner(self.flags)

    def run(self):
        """ Overriden in subclasses """
        raise NotImplementedError


class PrintDebugTest(ABCTest):
    def __init__(self, config=None):
        super(PrintDebugTest, self).__init__(config)

    def run(self):
        self.runner.run()
# todo implement seeded tests to compare to precomputed results
#test = PrintDebugTest()
#test.run()

"""
num_players = 2
hand_size = 2
candidate_counts = [3,2,2,2,1,3,2,2,2,1]
beliefs = np.array([[3,2,2,2,1,3,2,2,2,1],
 [3,2,2,2,1,3,2,2,2,1],
 [3,2,2,2,1,3,2,2,2,1],
 [3,2,2,2,1,3,2,2,2,1]])

re_marginalized = np.copy(beliefs)
for i, slot in enumerate(beliefs):
    #print('initial belief', beliefs)
    #print(np.sum(beliefs[(np.arange(num_players * hand_size) != i)], axis=0))
    #print(re_marginalized[i])
    re_marginalized[i] = np.sum(beliefs[(np.arange(num_players * hand_size) != i)], axis=0)
    #print(i, re_marginalized[i])
row_mask = [0 * hand_size + slot for slot in [0, 2, 3]]
print(f're_marginalized = \n{re_marginalized}')
re_marginalized[not row_mask] = 0
print(f're_marginalized with row mask = \n{re_marginalized}')
counts = np.array([3, 2, 2, 2, 0, 3, 2, 2, 2, 1])
mask = counts==0
print(mask)
re_marginalized[:, counts == 0] = 10000
print(f're_marginalized with impossible mask = \n{re_marginalized}')
# print([i for i in range(4) if i not in [0,2,3]])
#col_mask = [0,2]
#for r in col_mask:
#    re_marginalized[row_mask, r] = 1
# print(re_marginalized[row_mask, 0:3])
#print(f're_marginalized with row and col mask= \n{re_marginalized}')
"""

"""
from scipy.stats import rv_discrete
x = np.array([[3,2,2,2,1,3,2,2,2,1]])
px = np.array(x/np.sum(x))
num_ranks = 5
num_colors = 2
B = np.array([[3,2,2,2,1,3,2,2,2,1],
              [3,2,2,2,1,3,2,2,2,1],
              [3,2,2,2,1,3,2,2,2,1],
              [3,2,2,2,1,3,2,2,2,1],])
pB = np.array(B/np.sum(x, axis=1))
valB = np.array([[i for j in range(num_colors) for i in range(num_ranks)]])

for i_row, row in enumerate(B):
    x_i, px_i = B[i_row,:], np.array(B[i_row,:]/np.sum(B[i_row,:]))
    sample = rv_discrete(values=((x_i, px_i))).rvs(size=30)
    print(sample)
# print(valB)
"""
"""
candidate_count = np.array([3,2,2,2,1,3,2,2,2,1])
samples = np.array([
    [5, 7, 8, 0, 5, 0],
    [2, 4, 6, 7, 0, 8],
    [0, 0, 8, 2, 2, 5],
    [2, 7, 8, 1, 5, 1]]
)

sample = samples[:,0]
c = np.bincount(sample)
pad_width = (0, len(candidate_count) - len(c))
c = np.pad(c,pad_width=pad_width,mode='constant')
print(np.pad(c,pad_width=pad_width,mode='constant'))
print(candidate_count-c)
arr = candidate_count-c

if arr[arr < 0]:
    print('ASD:OIJA:SDJ')
else:
    print('nice')
"""