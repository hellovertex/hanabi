import os, sys
rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/')
sys.path.append(rel_path)

import agents.rainbow.run_experiment as xp
import vectorizer
import agent_player
import vectorizer_test


### Set up the environment
game_type = "Hanabi-Full"
num_players = 4
env = xp.create_environment(game_type=game_type, num_players=num_players)
# Setup Obs Stacker that keeps track of Observation for all agents ! Already includes logic for distinguishing the view between different agents
history_size = 1
obs_stacker = xp.create_obs_stacker(env,history_size=history_size)
observation_size = obs_stacker.observation_size()
obs_vectorizer = vectorizer.ObservationVectorizer(env)


agent_config = {
    "num_moves": 38,
    "vectorized_observation_shape": 1041,
    "players": 4,
}


# ### Set up the RL-Player, reload weights from most current trained model
# agent = "DQN"
agent = "Rainbow"
### create agent player from latest trained model
player = agent_player.RLPlayer(agent,env,observation_size,history_size)

# Create Vectorizer Objects to encode state and actions to Neural Network-digestable vectors
obs_vectorizer = vectorizer.ObservationVectorizer(env)
legalMovesVectorizer = vectorizer.LegalMovesVectorizer(env)

# Get Mock Observation - This all needs o happen on Server Side
mock_observation = vectorizer_test.get_mock_4pl_2nd_state()
obs_vectorizer.last_player_card_knowledge = [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]
mock_observation["last_moves"] = [{"type":"REVEAL_COLOR", "player":3, "hand_card_id":None, "color":"G", "RANK":None ,"target_offset": 1 , "scored":False, "information_token":False, "card_info_revealed":[1,3]}]
vectorized_obs_vectorizer = obs_vectorizer.vectorize_observation(mock_observation)
mock_observation["vectorized"] = vectorized_obs_vectorizer

legal_moves_as_int = legalMovesVectorizer.get_legal_moves_as_int(mock_observation["legal_moves"])
mock_observation["legal_moves_as_int"] = legal_moves_as_int

legal_moves_as_int_formated = legalMovesVectorizer.get_legal_moves_as_int_formated(legal_moves_as_int)
mock_observation["legal_moves_as_int_formated"] = legal_moves_as_int_formated

# Select Player Action
action = player.act(mock_observation)
print(action)

####### IMPORTANT #######
# Next we need to store the taken action in the vectorizer!
obs_vectorizer.last_player_action = action
