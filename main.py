import rl_env
from pyhanabi_env_wrapper import PyhanabiEnvWrapper

variant = "Hanabi-Full"
num_players = 5

pyhanabi_env = rl_env.make(environment_name=variant, num_players=num_players)
env = PyhanabiEnvWrapper(pyhanabi_env)




