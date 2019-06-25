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
import getopt
import rl_env
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent}


class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.agent_config = {'players': flags['players']}
        self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_class = AGENT_CLASSES[flags['agent_class']]
        self.game_state_wrappers = list()

    def run(self):
        """Run episodes."""
        rewards = []
        cards_played_correctly = []
        for episode in range(flags['num_episodes']):
            observations = self.environment.reset()
            agents = [self.agent_class(self.agent_config)
                      for _ in range(self.flags['players'])]
            done = False
            print("OBSERVATIONS LAST MOVES AFTER RESET")
            print(observations['player_observations'][0]['last_moves'])
            episode_reward = 0
            episode_correct_cards = 0

            while not done:
                for agent_id, agent in enumerate(agents):

                    observation = observations['player_observations'][agent_id]

                    #print("CURRENT PLAYER")
                    #print(observation['current_player'])
                    #print("CURRENT PLAYER OFFSET")
                    if len(observation['last_moves']) > 0:
                        print("PRINTING PLAYER")
                        print(observation['last_moves'][0].player())
                    #print(observation['current_player_offset'])
                    #print("LAST MOVES")
                    #print(observation['last_moves'])
                    # print(f"Player {observation['current_player']} to move:")
                    # print(observation['observed_hands'])
                    # print(observation.keys())
                    # print(observation['card_knowledge'])
                    action = agent.act(observation)
                    if observation['current_player'] == agent_id:

                        assert action is not None
                        current_player_action = action

                        break

                    else:
                        assert action is None

                # Make an environment step.
                print('Agent: {} action: {}'.format(observation['current_player'],
                                                    current_player_action))
                observations, reward, done, unused_info = self.environment.step(
                    current_player_action)
                episode_reward += reward
                if reward > 0:
                    episode_correct_cards += 1

            rewards.append(episode_reward)
            cards_played_correctly.append(episode_correct_cards)
            print('Running episode: %d' % episode)
            print('Max Reward: %.3f' % max(rewards))
            print(f'Max Cards played correctly: {max(cards_played_correctly)}')
        return rewards


if __name__ == "__main__":
    flags = {'players': 2, 'num_episodes': 1, 'agent_class': 'SimpleAgent'}
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
    runner = Runner(flags)
    runner.run()
