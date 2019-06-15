import rl_env
from pyhanabi_env_wrapper import PyhanabiEnvWrapper
# from tf_agents.environments import utils
import test

# game config
variant = "Hanabi-Full"
num_players = 5

# load and wrap environment, to use it with TF-Agent library
pyhanabi_env = rl_env.make(environment_name=variant, num_players=num_players)
env = PyhanabiEnvWrapper(pyhanabi_env)

# utils.validate_py_environment(env)
test.validate_py_environment(env)
# init dqn (rainbow) agent on env

# init policy agent on env





