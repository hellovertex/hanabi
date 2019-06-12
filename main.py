import rl_env
from pyhanabi_env_wrapper import PyhanabiEnvWrapper

# game config
variant = "Hanabi-Full"
num_players = 5

# load and wrap environment, to use it with TF-Agent library
pyhanabi_env = rl_env.make(environment_name=variant, num_players=num_players)
env = PyhanabiEnvWrapper(pyhanabi_env)

# init dqn (rainbow) agent on env

# init policy agent on env





