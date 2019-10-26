from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from custom_environment.pub_mdp import PubMDP
from custom_environment.pubmdp_env_wrapper import PubMDPWrapper
DEFAULT_CONFIG = {  # config for Hanabi-Small
        "colors": 2,
        "ranks": 5,
        "players": 2,
        "hand_size": 2,
        "max_information_tokens": 3,
        "max_life_tokens": 1,
        "observation_type": 1}

num_parallel_environments = 30


def load_hanabi_pub_mdp(game_config, policy_net=None):
    assert isinstance(game_config, dict)
    env = PubMDP(game_config, policy_net)
    if env is not None:
        return PubMDPWrapper(env)
    return None


policy_net = None  # todo get policy net
tf_env = tf_py_environment.TFPyEnvironment(
    parallel_py_environment.ParallelPyEnvironment(
        [lambda: load_hanabi_pub_mdp(DEFAULT_CONFIG, policy_net)] * num_parallel_environments)
)

# todo use ppo agent to train