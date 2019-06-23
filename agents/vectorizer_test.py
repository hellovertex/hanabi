import os
import sys

rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/')
sys.path.append(rel_path)

import numpy as np
import rainbow.run_experiment as xp
import vectorizer
import agent_player as ap
import rl_env
import tensorflow as tf
import agent_player

if __name__=="__main__":
    ### Set up the environment
    game_type = "Hanabi-Full"
    num_players = 4
    history_size = 1
    env = xp.create_environment(game_type=game_type, num_players=num_players)
    obs_stacker = xp.create_obs_stacker(env, history_size)

    agent_config = {
        "observation_size": 1041,
        "players": 4,
        "history_size": 1,
        "max_moves": env.num_moves()
    }

    #agent_test = agent_player.RLPlayer(agent_config)

    #sys.exit(0)

    # Exchange with agent player
    agent = xp.create_agent(env, obs_stacker)
    agent.eval_mode = True
    #agent_test.eval_mode = True
    actions_taken = 0

    # Setup vectorizer
    obs_vectorizer = vectorizer.ObservationVectorizer(env)
    moves_vectorizer = vectorizer.LegalMovesVectorizer(env)

    observations = env.reset()

    # print(observations)

    current_player, legal_moves, observation_vector  = xp.parse_observations(observations, env.num_moves(), obs_stacker)

    vec_obs = observations["player_observations"][current_player]
    own_vec = obs_vectorizer.vectorize_observation(vec_obs)

    wrong_indices = np.where(np.equal(observation_vector,own_vec)*1 == 0)
    # print(f"Wrong indices size: {wrong_indices[0].size}")

    moves_as_int = vec_obs["legal_moves_as_int"]
    # print(moves_as_int)

    own_moves_as_int = moves_vectorizer.get_legal_moves_as_int(vec_obs["legal_moves"])
    print("\n OWN LEGAL MOVES AS INT FORMATTED:")
    print(moves_vectorizer.get_legal_moves_as_int_formated(own_moves_as_int))
    print("\n")

    vec_obs["legal_moves_as_int_formated"] = moves_vectorizer.get_legal_moves_as_int_formated(own_moves_as_int)

    print(vec_obs['legal_moves_as_int_formated'])

    if wrong_indices[0].size > 0:
       print(f"Wrong Indices: {wrong_indices[0]}")
       for index in wrong_indices[0]:
           print(f"Index {index} was set to {own_vec[index]}")
           print(f"Should be: {observation_vector[index]}")
           print("================")
       sys.exit(0)

    #action = agent_test.act(vec_obs)
    #print(action)
    action = agent.begin_episode(current_player, legal_moves, observation_vector)

    print("\n==============================")
    print(f"OUTPUTED ACTION BY AGENT: {vec_obs['legal_moves'][np.where(np.equal(action,vec_obs['legal_moves_as_int']))[0][0]]}")
    print("================================\n")

    is_done = False
    total_reward = 0
    step_number = 0

    has_played = {current_player}

    LENIENT_SCORE = False

    # Keep track of per-player reward.
    reward_since_last_action = np.zeros(env.players)

    # simulate whole game
    while not is_done:
        print("\n====================")
        print(f"REAL MOVE TO BE TAKEN: {env.game.get_move(action)}")
        print("====================\n")
        observations, reward, is_done, _ = env.step(action.item())
        actions_taken+=1

        modified_reward = max(reward, 0) if LENIENT_SCORE else reward
        total_reward += modified_reward

        reward_since_last_action += modified_reward

        step_number += 1

        if is_done:
            print("Game is done")
            print(f"Steps taken {step_number}, Total reward: {total_reward}")
            break

        current_player, legal_moves, observation_vector_d  = xp.parse_observations(observations, env.num_moves(), obs_stacker)

        current_player_observation = observations["player_observations"][observations["current_player"]]

        observation_vector = current_player_observation["vectorized"]

        #print(np.equal(observation_vector, observation_vector_d))

        own_vec = obs_vectorizer.vectorize_observation(current_player_observation)

        wrong_indices = np.where(np.equal(observation_vector,own_vec)*1 == 0)
        # print(f"Wrong indices size: {wrong_indices[0].size}")

        moves_as_int = current_player_observation["legal_moves_as_int"]
        # print("\n=====================")
        # print("TEST LEGAL MOVES AS INT FORMATTER")
        # print("REAL MOVES AS INT")
        # print(moves_as_int)
        # print("OWN MOVES AS INT")
        own_moves_as_int = moves_vectorizer.get_legal_moves_as_int(current_player_observation["legal_moves"])
        print(own_moves_as_int)
        current_player_observation["legal_moves_as_int_formated"] =  moves_vectorizer.get_legal_moves_as_int_formated(own_moves_as_int)

        print("\n OWN LEGAL MOVES AS INT FORMATTED:")
        print(moves_vectorizer.get_legal_moves_as_int_formated(own_moves_as_int))
        print("\n")

        if wrong_indices[0].size > 0:
            print(f"Wrong Indices: {wrong_indices[0]}")
            for index in wrong_indices[0]:
                print(f"Index {index} was set to {own_vec[index]}")
                print(f"Should be: {observation_vector[index]}")
                print("================")
            print(current_player_observation["last_moves"])
            print(f"ACTIONS TAKEN = {actions_taken}")
            break

        #if len(wrong_indices[0]) > 0:
        #    break
        # print(len(np.where(np.equal(observation_vector,own_vec))))

        if current_player in has_played:

          #action = agent_test.act(current_player_observation)
          action = agent.step(reward_since_last_action[current_player],current_player, legal_moves, observation_vector)

          print("\n==============================")
          print(f"OUTPUTED ACTION BY AGENT: {current_player_observation['legal_moves'][np.where(np.equal(action,current_player_observation['legal_moves_as_int']))[0][0]]}")
          print("================================\n")

        else:
          # Each player begins the episode on their first turn (which may not be
          # the first move of the game).
          #action = agent_test.act(current_player_observation)
          action = agent.begin_episode(current_player, legal_moves,observation_vector)

          print("\n==============================")
          print(f"OUTPUTED ACTION BY AGENT: {current_player_observation['legal_moves'][np.where(np.equal(action,current_player_observation['legal_moves_as_int']))[0][0]]}")
          print("================================\n")

          has_played.add(current_player)

        # Reset this player's reward accumulator.
        reward_since_last_action[current_player] = 0

    agent.end_episode(reward_since_last_action)
