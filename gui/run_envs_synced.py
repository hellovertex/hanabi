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
from game_state_wrapper import GameStateWrapper
import pyhanabi_to_gui

def msg_deal_card(observation):
    """
    Input: pyahanabi observation object
    Output: notifyList message like:
    notifyList [{"type":"draw","who":0,"rank":3,"suit":3,"order":0},{"type":"draw","who":0,"rank":5,"suit":2,"order":1},{"type":"draw","who":0,"rank":4,"suit":4,"order":2},{"type":"draw","who":0,"rank":1,"suit":4,"order":3},{"type":"draw","who":0,"rank":1,"suit":2,"order":4},{"type":"draw","who":1,"rank":5,"suit":3,"order":5},{"type":"draw","who":1,"rank":4,"suit":2,"order":6},{"type":"draw","who":1,"rank":2,"suit":4,"order":7},{"type":"draw","who":1,"rank":1,"suit":2,"order":8},{"type":"draw","who":1,"rank":3,"suit":0,"order":9},{"type":"draw","who":2,"rank":-1,"suit":-1,"order":10},{"type":"draw","who":2,"rank":-1,"suit":-1,"order":11},{"type":"draw","who":2,"rank":-1,"suit":-1,"order":12},{"type":"draw","who":2,"rank":-1,"suit":-1,"order":13},{"type":"draw","who":2,"rank":-1,"suit":-1,"order":14},{"type":"status","clues":8,"score":0,"maxScore":25,"doubleDiscard":false},{"type":"text","text":"test goes first"},{"type":"turn","num":0,"who":2}]

    notifyList is used in the game_state_wrapper to assign the dealt cards to the players hands,
    see game_state_wrapper.GameStateWrapper.deal_cards(notify_msg)

    """




AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent}

# used for syncing game_state_wrappers



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
            # agent usernames - redundant but we keep it for readability
            usernames = [str(i) for i, _ in enumerate(agents)]
            game_config = {
                'num_total_players': 4,
                'life_tokens': 3,
                'info_tokens': 8,
                'deck_size': 50,
                'variant': 'Hanabi-Full',
                'colors': 5,
                'ranks': 5,
                'max_moves': 38
            }
            done = False
            episode_reward = 0
            episode_correct_cards = 0
            it = 0
            while not done:
                for agent_id, agent in enumerate(agents):
                    # get observation for pyhanabi env
                    observation = observations['player_observations'][agent_id]

                    if it == 0:
                        # create game_config for game_state_wrapper
                        agent_config = game_config
                        agent_config['username'] = str(agent_id)
                        # init game_state_wrapper for each agent
                        self.game_state_wrappers.append(GameStateWrapper(agent_config))

                        # inject init message
                        msg_init = pyhanabi_to_gui.create_init_message(observation, agent_id)
                        self.game_state_wrappers[agent_id].init_players(msg_init)

                        # inject notifyList message
                        msg_notifyList = pyhanabi_to_gui.create_notifyList_message(observation)
                        self.game_state_wrappers[agent_id].deal_cards(msg_notifyList)

                    print("VECTORIZED PYHANABI")
                    vectorized = observation['vectorized']
                    print(vectorized)
                    # get current observation for gui env
                    observation_gui = self.game_state_wrappers[agent_id].get_agent_observation()

                    # compare vectorized objects for each agent
                    # vectorized = observation['vectorized']
                    if it == 0:
                        vectorized_gui = observation_gui['vectorized']
                        print(observations)

                        print("VECTORIZED GUI")
                        print(vectorized_gui)
                        mask = vectorized_gui == vectorized
                        print(mask)
                        latest_false = 0
                        for i in range(len(mask)):
                            if mask[i] == False:
                                latest_false = i
                        print(latest_false)

                    # generate action
                    action = agent.act(observation)

                    # send json encoded action to all game_state_wrappers to update their internal state
                    # do this for each agent

                    #if action['action_type'] == 'REVEAL_COLOR':
                        # create notify message
                    #    notify_msg = None
                        # update game state wrappers
                    #    for i in range(len(agents)): game_state_wrappers[i].update_state(notify_msg)

                    if observation['current_player'] == agent_id:

                        assert action is not None
                        current_player_action = action
                        print("ACTION")
                        print(action)
                        break

                    else:
                        assert action is None

                    it += 1

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
    flags = {'players': 4, 'num_episodes': 1, 'agent_class': 'SimpleAgent'}
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
