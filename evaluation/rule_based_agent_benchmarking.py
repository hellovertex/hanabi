# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function

import sys
fp = str(__file__)[0:-len('evaluation/rule_based_agent_benchmarking.py')]
sys.path.insert(1, fp+'hanabi-learning-environment/')

import getopt
from hanabi_learning_environment import rl_env
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.rainbow_copy.dqn_agent import DQNAgent
from agents.rule_based_agent import RuleBasedAgent
import numpy as np
import matplotlib.pyplot as plt

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'DQN': DQNAgent, 'RuleBaseAgent': RuleBasedAgent}

colors = ["green", "blue", "red", "orange"]

class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.agent_config = {'players': flags['players']}
        self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_class = AGENT_CLASSES[flags['agent_class']]

    def run(self):
        """Run episodes or games."""
        scores = []
        for episode in range(flags['num_episodes']):
            observations = self.environment.reset()
            agents = [self.agent_class(self.agent_config['players'])
                      for _ in range(self.flags['players'])]
            done = False
            while not done:
                for agent_id, agent in enumerate(agents):

                    observation = observations['player_observations'][agent_id]

                    if observation['current_player'] == agent_id:
                        action = agent.act(observation)
                        assert action is not None
                        current_player_action = action
                    else:
                        action = agent.act(observation)
                        assert action is None
                # Make an environment step.
                observations, reward, done, unused_info = self.environment.step(
                    current_player_action)

            # get the score of the game
            score = 0
            for color, rank in observation['fireworks'].items():
                score = score + rank
            print("Score of game: {}".format(score))
            scores.append(score)

        print("Played {} games with {} players".format(flags['num_episodes'], flags['players']))
        scores = np.array(scores)
        print('Avg. Score is {} and Std. is {}'.format(scores.mean(), scores.std()))

        return scores


if __name__ == "__main__":
    flags = {'players': 2, 'num_episodes': 1000, 'agent_class': 'RuleBaseAgent'}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent_class='])

    if arguments:
        sys.exit('usage: rl_env_example.py [options]\n'
                 '--players       number of players in the game.\n'
                 '--num_episodes  number of game episodes to run.\n'
                 '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    all_scores = np.zeros((flags['num_episodes'], 4))

    for num_players in range(4):

        flags['players'] = num_players+2
        runner = Runner(flags)
        all_scores[:, num_players] = runner.run()

    # create bar chart showing avg. scores

    font = {'size': 26}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(5, 10))


    x_labels = range(2, 6)
    rects = plt.bar(x_labels, all_scores.mean(axis=0))
    plt.xticks(x_labels, x_labels)

    font = {'size': 26}
    plt.rc('font', **font)

    plt.xlabel("Number of Players")
    plt.ylabel("Avg. Score")
    plt.title('Benchmarking of Rule Based Agents, 1000 Trials'.format(flags['num_episodes']))
    plt.ylim(0, 25)

    for i, rect in enumerate(rects):

        rect.set_color(colors[i])
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                 str(height), ha='center', va='bottom')

    plt.show()
