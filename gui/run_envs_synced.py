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
import numpy as np
import utils

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

def last_false(mask):
    last_false = 0
    for i in range(len(mask)):
        if mask[i] == False:
            last_false = i
    return last_false

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
            vectorized_observations = list()  # list that stores vectorized observation for each agent
            while not done:
                for agent_id, agent in enumerate(agents):

                    # get observation from pyhanabi env
                    observation_pyhanabi = observations['player_observations'][agent_id]
                    # 1 pyhanabi observation contains 4 vectorized objects and each is computed individually for one agent
                    # you can see that, when you print out observation['vectorized'] for different agent_ids

                    # on reset, setup game_state_wraopers for each agent
                    if it == 0:
                        for i in range(len(agents)):
                            # create game_config for game_state_wrapper
                            agent_config = game_config
                            agent_config['username'] = str(i)
                            observation_tmp = observations['player_observations'][i]
                            # init game_state_wrapper for each agent
                            self.game_state_wrappers.append(GameStateWrapper(agent_config))

                            # inject init message
                            msg_init = pyhanabi_to_gui.create_init_message(observation_tmp, i)
                            self.game_state_wrappers[i].init_players(msg_init)

                            # inject notifyList message
                            msg_notifyList = pyhanabi_to_gui.create_notifyList_message(observation_tmp)
                            self.game_state_wrappers[i].deal_cards(msg_notifyList)


                            # get current observation for gui env
                            obs = self.game_state_wrappers[i].get_agent_observation()
                            vectorized_gui = obs['vectorized']
                            vectorized_observations.append(vectorized_gui)


                    # generate action
                    action = agent.act(observation_pyhanabi)

                    if observation_pyhanabi['current_player'] == agent_id:
                        observation_gui = self.game_state_wrappers[agent_id].get_agent_observation()
                        vectorized = np.array(observation_pyhanabi['vectorized'])
                        vectorized_gui = vectorized_observations[agent_id]

                        # compare the 2 vectorized objects
                        print('===========================================================')
                        print('===========================================================')
                        print('-----------------------COMPARISON--------------------------')
                        print('===========================================================')
                        print('===========================================================')
                        print("Vectorized objects of rl_env and gui are equal:")
                        equal = np.array_equal(vectorized, vectorized_gui)
                        print(equal)
                        print(f"SUM OF VEC PYHANABI = {sum(vectorized)}")
                        print(f"SUM OF VEC GUI = {sum(vectorized_gui)}")
                        if not equal:
                            last_false_idx = last_false(vectorized == vectorized_gui)
                            print(f"Last deviation at index: {last_false_idx}")
                            print(vectorized_gui == vectorized)
                            #print(observation_gui['observed_hands'])

                        print('===========================================================')
                        print('===========================================================')
                        print('-------------------END COMPARISON--------------------------')
                        print('===========================================================')
                        print('===========================================================')
                        assert action is not None
                        current_player_action = action
                        print("ACTION")
                        print(action)
                        break

                    else:
                        assert action is None

                    it += 1

                # Make an environment step.
                print('Agent: {} action: {}'.format(observation_pyhanabi['current_player'],
                                                    current_player_action))
                observations, reward, done, unused_info = self.environment.step(
                    current_player_action)

                # after step, synchronize gui env by using last_moves,
                if observation_pyhanabi['current_player'] == agent_id:

                    last_moves = observations['player_observations'][agent_id]['last_moves']
                    # send json encoded action to all game_state_wrappers to update their internal state
                    for i in range(len(agents)):
                        notify_msg = pyhanabi_to_gui.create_notify_message_from_last_move(self.game_state_wrappers[i],last_moves, agent_id)
                        self.game_state_wrappers[i].update_state(notify_msg)
                        # in case a card has been dealt, we need to update the game_state_wrappers as well
                        # we do this seperately because I forgot about it when encoding the notify messages :)
                        deal_msg = None
                        if len(last_moves) > 0 and last_moves[0].move().type() == utils.HanabiMoveType.DEAL:
                            # we have to use last_moves of different agent in order to see color and rank of drawn card
                            idx_next = (agent_id + 1) % observation_pyhanabi['num_players']
                            last_moves = observations['player_observations'][idx_next]['last_moves']
                            deal_msg = pyhanabi_to_gui.create_notify_message_deal(self.game_state_wrappers[i],last_moves, agent_id)
                        if deal_msg != None:
                            self.game_state_wrappers[i].update_state(deal_msg)

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
