import time
import numpy as np
import random
import os
import tensorflow as tf

import model
import game
import population
import util
from hanabi_learning_environment import pyhanabi
# todo change env creation inside Game.__init__ method
# todo call network inside bad_agent.step
env_config = {
            "colors":
                5,
            "ranks":
                5,
            "players":
                2,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
            'use_custom_rewards':
                True,
            'open_hands':
                False
        }

obs_size, num_actions = game.load_specs(env_config)
rewards_weights_base, _, model_config_base, random_attributes = util.get_confs()
root_dir = './training'

n_to_evolve = 5
mutation_prob = 0.5
ts_per_epoch = 600000
evaluate_last_updates = 50
population_name = 'pbt_test'
save_every = 20

# create pool
num_envs = model_config_base['nenvs']
tf.reset_default_graph()
sess = tf.Session()

# starts game with bad_agent.Player instances
game = game.Game(population_size=2,
                 num_envs=num_envs, env_config=env_config, wait_rewards=True)


population = population.Population(num_actions=num_actions,
                                   obs_size=obs_size,
                                   num_players=2,
                                   sess=sess,
                                   num_models=8,
                                   model_config_base=model_config_base,
                                   rewards_config_base=rewards_weights_base,
                                   random_attributes=random_attributes,
                                   folder=root_dir,
                                   name=population_name)

for epoch in range(2):
    population.run_epoch(game=game,
                         timesteps=1000,
                         n_to_evolve=5,
                         mutation_prob=.5,
                         summary_every=20000)
