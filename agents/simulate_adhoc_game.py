import os
import sys

rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/')
sys.path.append(rel_path)

import numpy as np
import rainbow.run_experiment as xp
# import pyhanabi_vectorizer as vectorizer
# import vectorizer
import rl_env
import tensorflow as tf
from adhoc_player import AdHocRLPlayer
from tf_agents_lib import pyhanabi_env_wrapper
from tf_agents.environments import tf_py_environment


def get_legal_moves_as_int_formatted_from_mask(mask):
    """ Replaces 0 with -inf and 1 with 0, s.t. evaluation can be performed on dopamine-DQN (i.e. rainbow) as well """
    # replace 0 with neg inf
    tmp = np.where(mask == 0, -np.inf, mask)
    # replace 1 with 0
    return np.where(tmp == 1, 0, tmp)


def run(players):
    # string formatted agent classes
    playing_agents = players

    # Game-simulation Parameters
    max_reward = 0
    eval_episodes = 1
    LENIENT_SCORE = False

    game_type = "Hanabi-Full"
    num_players = 4
    history_size = 1
    observation_size = 1041

    # Simulation objects
    env = xp.create_environment(game_type=game_type, num_players=num_players)
    max_moves = env.num_moves()

    # add wrapped && TfEnv here
    wrapped_env = pyhanabi_env_wrapper.PyhanabiEnvWrapper(env)
    tf_env = tf_py_environment.TFPyEnvironment(wrapped_env)

    obs_stacker = xp.create_obs_stacker(env, history_size)

    # debug purposes
    with tf.compat.v1.Session() as sess:
        obs = tf_env.reset()
        print("SESS_RUN")
        print(sess.run(obs))

    """ print("DEUBG")
        print(sess.run(obs.step_type))  # 0,1,2 for FIRST, MID, LAST
        print(sess.run(obs.observation['mask']))
        print("OBTAINING ENV OBSERVATIONS")
        print(tf_env.pyenv.envs[0]._make_observation_all_players() )
        # obs = sess.run(tf_env.step(int(0)))  # does not work as expected action type doesnt match
        print("DEUBG")
        print(sess.run(obs.step_type))  # 0,1,2 for FIRST, MID, LAST
        print("OBTAINING ENV OBSERVATIONS")
        print(tf_env.pyenv.envs[0]._make_observation_all_players())
    """
    agents = [
        AdHocRLPlayer(observation_size, num_players, history_size, max_moves, player) for player in playing_agents
    ]

    for agent in agents:
        if hasattr(agent, 'eval_mode'):
            agent.eval_mode = True

    # Game Loop: Simulate # eval_episodes independent games
    for ep in range(eval_episodes):

        is_done = False
        total_reward = 0
        step_number = 0

        # Keep track of per-player reward.
        reward_since_last_action = np.zeros(env.players)

        # for evaluation with tf_agent library
        with tf.compat.v1.Session() as sess:
            # reset environment
            tf_observations = tf_env.reset()
            sess.run(tf_observations)
            env = tf_env.pyenv.envs[0]

            # simulate whole game
            while not is_done:

                # for evaluation with dopamine
                py_observations = env._make_observation_all_players()
                current_player, legal_moves, observation_vector = xp.parse_observations(py_observations, env.num_moves(),
                                                                                        obs_stacker)
                current_player_observation = py_observations["player_observations"][current_player]

                # get mask from tf environment and compute legal moves as int
                mask = sess.run(tf_observations.observation['mask'])
                # todo compare legal moves as int, since we got move is legal assertion error
                current_player_observation["legal_moves_as_int_formated"] = get_legal_moves_as_int_formatted_from_mask(mask)

                # action sampling [rainbow, tf_agent]
                action = agents[current_player].act(current_player_observation)
                # todo tf_agents sample action from policy using TimeStep

                # todo step tf environment instead
                observations, reward, is_done, _ = env.step(action.item())

                modified_reward = max(reward, 0) if LENIENT_SCORE else reward
                total_reward += modified_reward

                reward_since_last_action += modified_reward

                step_number += 1

                if is_done:
                    print("Game is done")
                    print(f"Steps taken {step_number}, Total reward: {total_reward}")
                    if max_reward < total_reward:
                        max_reward = total_reward

    print(f"Max episode reached over {eval_episodes} games: {max_reward}")


if __name__ == "__main__":
    players = ['custom_dis_punish', 'custom_dis_punish', '10kit', '10kit']

    run(players)

