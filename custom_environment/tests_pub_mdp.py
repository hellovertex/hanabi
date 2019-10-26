from hanabi_learning_environment import rl_env
from agents.simple_agent import SimpleAgent
from custom_environment.pub_mdp import  PubMDP
from custom_environment.pubmdp_env_wrapper import PubMDPWrapper
import os
import timeit
import time
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
    #self.environment = rl_env.HanabiEnv(flags['game_config'])
    self.environment = PubMDP(flags['game_config'])
    self.wrapped_env = PubMDPWrapper(self.environment)
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
      print(self.wrapped_env.time_step_spec())
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          augmented_observation = self.pub_mdp.augment_observation(observation)
          action = agent.act(observation)
          if observation['current_player'] == agent_id:
            # print(observations['s_bad'])
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
"""
arr = np.tile([3,2,2,2,1], (4,2))
et0 = time.time()
def enum_time():
    return [(i, slot) for i, slot in enumerate(arr)]
et1 = time.time()

et = et1 -et0

nt0 = time.time()
def ndenum_time():
    return [(i, slot) for i,slot in np.ndenumerate(arr)]
nt1 = time.time()
nt = nt1 - nt0
print(f'ndenum was {et/nt} times faster')

A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
np.add.at(Z, I, 1)
print(I)
print(Z)
"""
"""

start = time.time()
# have 20k samples, now iterate cols
for idx in range(sampled.shape[1]):
    sample = sampled[:, idx]
    sample_card_counts = np.bincount(sample, minlength=len(candidate_counts))
    reduced_counts = candidate_counts - sample_card_counts
    if reduced_counts[reduced_counts < 0].size == 0:
        pass
    else:
        continue  # start next iteration, as this sample was inconsistent
    for shape, val in np.ndenumerate(sample):
        if hint_mask[shape[0], val] == 0:
            continue
    # if we reach this code, the sample is consistent
    consistent_samples.append(idx)
    end = time.time()
tp = end - start
"""

"""
# generate a 4 by 20000 matrix and time for loop and numpy it
x = np.array([card_index for card_index in range(5 * 2)])
B = np.array([[3,2,2,2,1,3,2,2,2,1],
              [3,2,2,2,1,3,2,2,2,1],
              [3,2,2,2,1,3,2,2,2,1],
              [3,2,2,2,1,3,2,2,2,1],])
hint_mask = np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1],
                      [1,1,1,1,1,1,1,1,1,1]])
candidate_counts = np.array([3,2,2,2,1,3,2,2,2,1])
px = B / np.sum(B, axis=1, keepdims=True)
num_samples = 20000
sampled = np.zeros(shape=(4, num_samples), dtype=np.int)
consistent_samples = list()
start = time.time()
for (i_row, _), row in np.ndenumerate(B):
    sampled[i_row,] = rv_discrete(values=(x, px[i_row, :])).rvs(size=num_samples)

idx = 0
for sample in sampled.T:
    sample_card_counts = np.bincount(sample, minlength=len(candidate_counts))
    reduced_counts = candidate_counts - sample_card_counts
    if reduced_counts[reduced_counts < 0].size == 0:
        pass
    else:
        continue
    for shape, val in np.ndenumerate(sample):
        if hint_mask[shape[0], val] == 0:
            continue
    consistent_samples.append(idx)
    if len(consistent_samples)==3000:
        break
    idx += 1
print(len(sampled[:,consistent_samples].T))
end = time.time()
tn = end - start
#print(f'coliteration was {tp/tn} times faster')
print(f' took {tn*1000} milliseconds')
"""
