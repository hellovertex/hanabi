import os
import sys

rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/')
sys.path.append(rel_path)
sys.path.append('reinforce_agent')

from agents.tf_agent_adhoc_reinforce_player import ReinforceTfAgentAdHocPlayer
from tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper
import rl_env


if __name__=="__main__":

    # Game-simulation Parameters
    max_reward = 0
    total_reward_over_all_ep = 0
    eval_episodes = 50
    LENIENT_SCORE = False

    game_type = "Hanabi-Full"
    num_players = 4
    history_size = 1
    observation_size = 1041

    # Simulation objects
    pyhanabi_env = rl_env.make(environment_name=game_type, num_players=num_players)
    py_env = PyhanabiEnvWrapper(pyhanabi_env)
    #moves_vectorizer = vectorizer.LegalMovesVectorizer(pyhanabi_env)
    max_moves = pyhanabi_env.num_moves()

    ### Reinforcement Learning Agent Player
    reinforce_agent = ReinforceTfAgentAdHocPlayer(
        'reinforce_policy_old',
        game_type = "Hanabi-Full",
        num_players = 4,

    )
    agents = [
                reinforce_agent,
                reinforce_agent,
                reinforce_agent,
                reinforce_agent
              ]

    for agent in agents:
        agent.eval_mode = True

    # Game Loop: Simulate # eval_episodes independent games
    for ep in range(eval_episodes):

        time_step = py_env.reset()

        total_reward = 0
        step_number = 0
        reward_since_last_action = 0

        # simulate whole game
        while not time_step.is_last():

            current_player = py_env._env.state.cur_player()
            action = agents[current_player].act(time_step)

            time_step = py_env.step(action)
            reward = time_step.reward

            modified_reward = max(reward, 0) if LENIENT_SCORE else reward
            total_reward += modified_reward

            reward_since_last_action += modified_reward
            if modified_reward >= 0:
                total_reward_over_all_ep += modified_reward

            step_number += 1

            if time_step.is_last():
                print("Game is done")
                print(f"Steps taken {step_number}, Total reward: {total_reward}")
                if max_reward < total_reward:
                    max_reward = total_reward

    print(f"Average reward over all actions: {total_reward_over_all_ep / eval_episodes}")
    print(f"Max episode reached over {eval_episodes} games: {max_reward}")