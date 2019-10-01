from absl import app
from absl import flags
import os
import tensorflow as tf
from tf_agents.policies import py_tf_policy

from hanabi_learning_environment import rl_env
from training.tf_agents_lib.pyhanabi_env_wrapper import PyhanabiEnvWrapper
from evaluation.tf_agent_adhoc_player_ppo import PPOTfAgentAdHocPlayer
from tf_agents.environments import tf_py_environment
from training.tf_agents_lib.masked_networks import MaskedActorDistributionNetwork
from training.tf_agents_lib.masked_networks import MaskedValueNetwork
from tf_agents.agents.ppo.ppo_agent import PPOAgent

flags.DEFINE_string('root_dir', os.getenv('UNDEFINED'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_episodes', 1,
                     'Number of games to be played')
flags.DEFINE_integer('num_players', 4,
                     'Number of agents playing')
flags.DEFINE_string('game_type', 'Hanabi-Small',
                    'Determines which type of HanabiGame instance is created')
FLAGS = flags.FLAGS


def load_networks(tf_env):
    """ Initializes actor and value networks for ppo agent """
    actor_net = MaskedActorDistributionNetwork(  # set up actor network as trained before
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(150, 75)
    )
    value_net = MaskedValueNetwork(  # set up value network as trained before
        tf_env.observation_spec(), fc_layer_params=(150, 75)
    )
    return actor_net, value_net


class TrainedPPOAgent:
    def __init__(self, train_dir):
        # --- Tf session --- #
        tf.reset_default_graph()
        self.sess = tf.Session()
        tf.compat.v1.enable_resource_variables()

        # --- Environment Stub--- #
        env = rl_env.make(environment_name=FLAGS.game_type, num_players=FLAGS.num_players)
        wrapped_env = PyhanabiEnvWrapper(env)
        tf_env = tf_py_environment.TFPyEnvironment(wrapped_env)

        with self.sess.as_default():
            # --- Init Networks --- #
            actor_net, value_net = load_networks(tf_env)

            # --- Init agent --- #
            agent = PPOAgent(  # set up ppo agent with tf_agents
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                actor_net=actor_net,
                value_net=value_net,
                train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
                normalize_observations=False
            )

            # --- init policy --- #
            self.policy = py_tf_policy.PyTFPolicy(agent.policy)
            self.policy.initialize(None)
            # --- restore from checkpoint --- #
            self.policy.restore(policy_dir=train_dir, assert_consumed=False)

            # Run tf graph
            self.sess.run(agent.initialize())

    def act(self, observation):
        with self.sess.as_default():
            policy_step = self.policy.action(observation)
            return policy_step.action


def main(_):
    """ Runs {FLAGS.num_episodes} games of Hanabi-small using a trained ppo agent"""

    # --- Load ppo agent with trained policy --- #
    agent = TrainedPPOAgent(train_dir=FLAGS.root_dir)

    # --- Create Environment for evaluation --- #
    env = rl_env.make(environment_name=FLAGS.game_type, num_players=FLAGS.num_players)
    eval_env = PyhanabiEnvWrapper(env)

    # --- Run single game for inspection --- #
    i = 1
    sum_rewards = 0
    time_step = eval_env.reset()

    for i in range(FLAGS.num_episodes):
        while not time_step.is_last():
            # act
            action = agent.act(time_step)
            # step
            time_step = eval_env.step(action)
            print(f'Got reward {time_step.reward} at turn {i}')
            sum_rewards += time_step.reward
            i += 1

    if time_step.is_last():
        print(f'Game ended at turn {i-1}')

    print(f'total reward was {sum_rewards}')


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
