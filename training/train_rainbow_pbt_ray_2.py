"""Different approach to running several environments in parallel than train_rainbow_pbt.py using the Ray package:
Every worker is a Ray actor with its own tensorflow instance. This costs extra memory but keeps the variables apart
without the need for registering the variables.
Copy functions are different, of course.
"""
### IMPORTS
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
sys.path.insert(0,"/home/ma/uni/hanabi/hanabi")

from absl import app
from absl import flags

import numpy as np

import sys
from hanabi_learning_environment.agents.rainbow.third_party.dopamine import logger
from hanabi_learning_environment.agents.rainbow.third_party.dopamine import iteration_statistics
# import hanabi_learning_environment.agents.rainbow
import hanabi_learning_environment.agents.rainbow.run_experiment as run_experiment
# import hanabi_learning_environment.rl_env_custom as rl_env

import pickle
import shutil
import pandas as pd
import time
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib.lines import Line2D
import ray
#tensorflow is imported inside Ray actor to avoid problems with global tensorflow state

### FLAGS
flags.DEFINE_string('base_dir', str(os.path.dirname(__file__)) + '/trained/PBTRainbow',
                    'Base directory to host all required sub-directories. '
                    'Path for logs and checkpoints')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Directory where checkpoint files should be saved.')
flags.DEFINE_string('logging_dir', 'logs',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('game_type', 'Hanabi-Full',
                    'Hanabi-Full or Hanabi-Small, etc.')
# FLAGS = flags.FLAGS


### PBT worker functionality
@ray.remote
class Member(object):
    """Member of population, wraps a parameterized training routine"""
    def __init__(self, params, idx):
        """
        Args:
            params: all hyperparameters, including the immutable ones
            idx: idx of member in population, for convenience
        """
        import tensorflow as tf #tensorflow is imported inside Ray actor to avoid problems with global tensorflow
        self.tf = tf
        # state and makes it easier to transfer weights: https://ray.readthedocs.io/en/latest/using-ray-with-tensorflow.html
        self.id = str(idx).zfill(3)  # identifier number string
        self.ckpt_dir = os.path.join(FLAGS.base_dir, FLAGS.checkpoint_dir, self.id)
        self.params = params  # members parameters
        self.parent_idx = None  # save parent idx before every pbt step
        # Initialize the environment.
        self.environment = run_experiment.create_environment(game_type=params["game_type"],
                                                             num_players=params["num_players"],
                                                             )
        # Initialize the Logger object.
        self.logger = logger.Logger(os.path.join(FLAGS.base_dir, FLAGS.logging_dir, self.id))
        # Initialize the observation stacker.
        self.obs_stacker = run_experiment.create_obs_stacker(self.environment)
        # Initialize the agent, no scope needed
        self.agent = run_experiment.create_agent(self.environment, self.obs_stacker,
                                                 agent_type=self.params["agent_type"],
                                                 tf_session=None, #new session will be created, private to ray actor
                                                 config = self.params["rainbow_config"])
        self.train_step = 0
        self.statistics = []  # save statistics after every pbt step here (so when copying the agent from a
        # member the training progress will not be lost
        self.pbt_step, self.checkpointer = (  # pbt_step is current pbt_step
            run_experiment.initialize_checkpointing(self.agent,
                                                    self.logger,
                                                    self.ckpt_dir))

    def train(self):
        """Perform a training phase of given number of training iterations.
        self.params["training_steps"] corresponds to length of "glued-together" training episodes s.t. they have at
        least training_steps number of game steps, not overall episodes!"""
        start_time = time.time()

        statistics = iteration_statistics.IterationStatistics()

        # First perform the training phase, during which the agent learns.
        self.agent.eval_mode = False
        number_steps, sum_returns, num_episodes = (
            run_experiment.run_one_phase(self.agent, self.environment, self.obs_stacker, self.params["training_steps"],
                                         statistics, run_mode_str='train'))
        #custom rewards, episode lengths and returns are directly added to statistics inside run_one_phase
        time_delta = time.time() - start_time
        self.tf.logging.info('Average training steps per second: %.2f',
                        number_steps / time_delta)

        average_return = sum_returns / num_episodes
        self.tf.logging.info('Average per episode return: %.2f', average_return)
        return statistics.data_lists

    def eval(self):
        """Perform an evaluation phase of given number of evaluation iterations."""
        start_time = time.time()
        statistics = iteration_statistics.IterationStatistics()
        self.agent.eval_mode = True

        e_lengths = []
        e_returns = []
        e_reward_info = {"environment_reward": [],
                       "hint_reward": [],
                       "play_reward": [],
                       "discard_reward": []}
        # Collect episode data for all games.
        for _ in range(self.params["num_evaluation_games"]):
            length, reward, info = run_experiment.run_one_episode(self.agent, self.environment, self.obs_stacker)
            e_lengths.append(length)
            e_returns.append(reward)
            for key in info.keys():  # aggregate step's custom reward info
                e_reward_info[key].append(info[key])

        #Agglomerate values and add to statistics
        statistics.append({
            'eval_episode_lengths': e_lengths,
            'eval_episode_returns': e_returns,
            **dict([(f"eval_episode_{key}s", e_reward_info[key]) for key in
                   e_reward_info.keys()])})

        self.tf.logging.info('Average eval. episode length: %.2f  Return: %.2f', e_lengths, e_returns)

        time_delta = time.time() - start_time
        self.tf.logging.info(f'Evaluating took {time_delta} seconds')
        return statistics.data_lists

    def stepEval(self):
        """For convenience: Training step followed by evaluation of learned model.
        A bit more convenient as checkpointing works cleaner for this at the moment, needs to be improved for
        individual train() and eval() functions.

        Returns: dictionary containing train and eval statistics"""
        # training and evaluation step (run_one_iteration does both with params suitably set)
        train_stats = []
        for _ in range(self.params["num_iterations"]):
            train_stats.append(self.train()) #todo: make all function calls remote?
            self.train_step += self.params["training_steps"]
        eval_stat = self.eval()
        self.pbt_step += 1
        return self.augment_stats(train_stats, eval_stat)

    def acc_dicts(self, list_of_dicts):
        """Accumulate list of dicts into individual dict, assuming all have the same keys as first dictionary in list.
        Args:
            list_of_dicts: list of dictionaries

        Returns: Dict with same keys, for each key lost of all corresponding entries.

        """
        final_dict = {}
        for key in list_of_dicts[0].keys():
            final_dict[key] = [d[key] for d in list_of_dicts]
        return final_dict

    def augment_stats(self, train_stats, eval_stat):
        """append current training statistics and values of mutables together with some overhead information to
        self.statistics
        """
        train_stat = self.acc_dicts(train_stats) #merge train and eval iteration statistics into one each
        stats = {**train_stat, **eval_stat} #combine train and eval stat
        #clean up: lists with single entries to single entry in dict
        for key, value in stats.items():
            if type(value) == list and len(value)==1:
                stats[key] = value[0]
        #append overhead business  to statistics so every entry in statistics is completely interpretable by itself
        stats["id"] = self.id
        stats["parent_idx"] = self.parent_idx
        stats["pbt_step"] = self.pbt_step
        stats["train_step"] = self.train_step
        #look up all current values of mutable variables
        current_mutable_vals = dict((par_name, self.params[par_name]) for par_name in self.params["pbt_mutables"])
        #append everything together as one dictionary to self.statistics
        return {**stats, **current_mutable_vals}


    def change_param(self, param, value):
        self.params[param] = value #overwrite value
        if param in ["w_hint", "w_play", "w_disc"]: #these need to be set inside environment to become effective
            setattr(self.environment, param, value)

    def export_agent(self):
        """Export agent in yet to determined format so it can play with others inside the GUI.
        TODO: implement ;)
        """

    def save_ckpt(self):
        """Save entire member object by pickling crucial fields and using tf.saver functionality to save agent.

        member-specific information is saved in ckpt_dir/memberinfo.pickle, agent is saved in ckpt_dir/agent/
        Agent can be loaded by calling .load_ckpt() after initialization.
        """
        #save parameters, parent_idx and statistics "manually"
        pickle.dump((self.params, self.train_step),
                    open(os.path.join(self.ckpt_dir, f"memberinfo.{self.pbt_step}"), "wb"))
        #save agent using agent's checkpointer
        run_experiment.checkpoint_experiment(self.checkpointer, self.agent, self.logger,
                                  self.pbt_step, self.ckpt_dir, checkpoint_every_n=1)
        #checkpoint_experiment retains the previous 3 checkpoints and deletes all older ones (inside the definition
        # of DQNAgent tf.train.Saver(max_to_keep=3). Now do the same cleanup for memberinfo
        try:
            for filename in os.listdir(self.ckpt_dir):
                if filename.startswith("memberinfo") and int(filename.split('.')[-1]) <= self.pbt_step - 30:
                    os.remove(os.path.join(self.ckpt_dir, filename))
        except Exception as e: #suppress and IO errors from deleting files, just print the message
            print(e)
        #todo: check if tf meta files clutter directory

    def load_ckpt(self):
        """Load the pickled hyperparameters. Tensorflow model parameters were already imported from checkpoint after
        "empty" member was initialized utilizing the checkpoint system. This method only checks whether
        hyperparameter epoch matches model param epoch.
        """
        try: #load hyperparameters
            (self.params, self.train_step) = pickle.load(
                open(os.path.join(self.ckpt_dir, f"memberinfo.{self.pbt_step}"), "rb"))
        except FileNotFoundError:
            raise Exception("No ckpt file found for Member {self.id} at pbt_step {self.pbt_step}")
        return self.pbt_step

    def get_state(self):
        # self.id and self.ckpt_dir  self.environment, self.logger and self.obs_stacker remain the same
        var_dict = {[(var.name, self.agent._sess.run(var).copy()) for var in self.tf.global_variables()]}
        return(var_dict, self.params, self.train_step)

    def set_state(self, var_dict, params, train_step):
        self.params = params
        self.train_step = train_step
        for var in self.tf.global_variables():
            var.load(var_dict[var.name], self.agent._sess)  # load it, sessions of old and new agent should be the same

    def mutate(self):
        """Mutates parameters in-place according to PBT experiments mutation hyperparameters."""
        for param in self.params["pbt_mutables"]:
            if np.random.uniform(0, 1) < self.params["mutprob"]:
                print(f"mutating {param} of member {self.id}")
                # value = member.params["rainbow_config"][param] #modifies in-place but we are going to overwrite it
                value = self.params[param]
                value += value * np.random.uniform(-1, 1) * self.params["mutstren"]
                if param == "gamma":
                    value = np.clip(value, 0, 1)
            self.change_param(param, value) #self change_param encapsulates information how to effectively change
            # a parameter


### PBT evolutionary functionality
def exploit(member, good_population):
    """Overwrites hyperparams and parameters of agent in member with a randomly chosen member of good_population"""
    newMember = good_population[np.random.choice(range(len(good_population)))]
    print(f"overwriting member {member.id} with member {newMember.id}")
    #get weights from newMember
    weights = newMember.get_weights()
    #write into member
    member.set_weights(weights)

#### various convenience and checkpointing fns
def load_or_create_population(pbt_popsize, def_params):
    ray.init() #memory=2*1e9) #initialize ray
    #initializing the agents creates an empty ckpt directory, so we have to check before whether checkpoint dir exists
    ckpt_dir_exists = os.path.isdir(os.path.join(FLAGS.base_dir, FLAGS.checkpoint_dir))

    #initialize members in population, init_sess_cars_ckpting initializes all tensorflow-related stuff
    population = []
    for i in range(pbt_popsize):
        population.append(Member.remote(def_params, i)) #.remote creates an actor instance of Member

    if False: #ckpt_dir_exists: #load parameters and parameters of each member from checkpoint
        ckpt_steps = set(ray.get([m.load_ckpt.remote() for m in population]))
        if len(ckpt_steps) > 1:
            raise Exception(f"inconsistent member checkpoints found: {ckpt_steps}")
        else:
            startstep = ckpt_steps.pop()
    else: #start from fresh
        #members are good to run
        startstep = 0
    return(population, startstep)

def save_population(population, stats):
    """Saves all members by calling their checkpointers and also dumps pbt statistics as df to disk.

    Returns: statistics of pbt experiment as df
    """
    for member in population:
        member.save_ckpt.remote()
    stats_df = pd.DataFrame.from_records(stats) #make df from list of dicts
    try:
        pickle.dump((stats_df, def_params), open(os.path.join(FLAGS.base_dir,"pbt_stats.pickle"), 'wb'))
    except Exception as e:
        print("writing statistics failed:")
        print(e)
    return stats_df, stats

def plot_statistics(stats, def_params=None):
    """Convenience function to plot statistics of PBT run

    Args:
        stats_ref: either pandas DataFrame or filepath to pickle file
    """
    if type(stats) == 'str' and os.path.isfile(stats):
        (stats, def_params) = pickle.load(open(stats, 'rb'))
    if not type(stats) == pd.DataFrame:
        raise TypeError("Statistics are no Pandas DataFrame")
    if def_params == None:
        raise Exception("Default parameters missing")

    #stats now is assumed to be a correctly formatted DataFrame
    #aggregate lists into mean and std for given axes
    for measure in ['train_episode_lengths', 'train_episode_returns',
                   'train_episode_environment_reward', 'train_episode_hint_reward',
                   'train_episode_play_reward', 'train_episode_discard_reward',
                   'eval_episode_lengths', 'eval_episode_returns',
                   'eval_episode_environment_rewards', 'eval_episode_hint_rewards',
                   'eval_episode_play_rewards', 'eval_episode_discard_rewards']:
        stats[f"mean_{measure}"] = stats[measure].apply(np.mean)
        stats[f"std_{measure}"] = stats[measure].apply(np.std)
    #get ids and corresponding colors globally to keep colors consistent across plots
    ids = list(set(stats["id"]))
    n_membs = len(ids)
    # colors = cm.get_cmap("gist_rainbow", n_membs)
    marker_style = dict(linestyle=':', marker='o',
                        markersize=10, markerfacecoloralt='white')

    #evaluation accuracy of all models
    fig, ax = plt.subplots()
    ax.set_title("Members' eval. accuracy")
    for i, id in enumerate(ids): #iterate over members
        id_df = stats[stats.id==id] #pick subset of data
        ax.plot(id_df.train_step, id_df.mean_eval_episode_returns, color=f'C{i}', label='member '+id, alpha=0.8)
        ax.fill_between(id_df.train_step,
                        id_df.mean_eval_episode_returns - id_df.std_eval_episode_returns,
                        id_df.mean_eval_episode_returns + id_df.std_eval_episode_returns,
                        color=f'C{i}', alpha=0.2)
    ax.legend()
    ax.set_ylabel(f'Accuracy averaged over {def_params["num_evaluation_games"]} evaluation epochs')
    ax.set_xlabel("Train Steps")

    #values of mutables over time
    fig, ax = plt.subplots()
    ax.set_title("Members' mutable variable values")
    for i, id in enumerate(ids): #iterate over members
        id_df = stats[stats.id==id] #pick subset of data
        for mutable, fillstyle in zip(def_params["pbt_mutables"],Line2D.fillStyles):
            ax.plot(id_df.train_step, id_df[mutable], color=f'C{i}', label=f'member {id}: {mutable}',
                    fillstyle = fillstyle, **marker_style)
    ax.legend()
    ax.set_ylabel("Value of mutable variables")
    ax.set_xlabel("Train Step")

    #winning model average return, eval episode lengths and eval episode returns
    winning_model_id = stats.iloc[stats.mean_eval_episode_returns.idxmax()].id
    id_df = stats[stats.id == winning_model_id]  # pick subset of data
    measures = ["eval_episode_lengths", "eval_episode_returns"]
    fig, ax = plt.subplots()
    ax.set_title("Winning Model")
    for measure,fillstyle in zip(measures,Line2D.fillStyles):
        ax.plot(id_df.train_step, id_df[f"mean_{measure}"], color=f'C{ids.index(winning_model_id)}', label=f'member '
                f'{winning_model_id}: {measure}', fillstyle = fillstyle, **marker_style)
        ax.fill_between(id_df.train_step,
                        id_df[f"mean_{measure}"] - id_df[f"std_{measure}"],
                        id_df[f"mean_{measure}"] + id_df[f"mean_{measure}"],
                        color=f'C{i}', alpha=0.2)
    ax.legend()
    ax.set_ylabel("Performance measure")
    ax.set_xlabel("Train Step")

#### PBT hyperparameters and default model params
#in "The Hanabi Challenge" paper, total number of training samples, i.e. steps from the environment, is limited to 1e8
pbt_steps = 5 # 4000
pbt_popsize = 1
pbt_mutprob = 0.25  # probability for a parameter to mutate
pbt_mutstren = 0.5  # size of interval around a parameter's current value that new value is sampled from, relative to current value
pbt_survivalrate = 0.75  # percentage of members of population to survive, rest will be discarded, replaced and mutated
pbt_discardN = int(pbt_popsize * (1 - pbt_survivalrate))  # for convenience


### define default config for currently trained agent and decide which parameters can be mutated
# so far using rl_env.make encapsulation of config when creating the environment
def_params = {"game_type": 'Hanabi-Full',  # environment parameters
              "agent_type": 'Rainbow',
              "num_players": 2,
              "training_steps": 500,  # schedule for training and eval step for particular set of hyperparameters
              "num_iterations": 1,
              "num_evaluation_games": 10,
              "mutprob": pbt_mutprob, #probability of a parameter to mutate
              "mustren": pbt_mutstren, # mutation strength (see above)
              "rainbow_config": {
                  "num_atoms": 51,                  #number of bins that constitute support of distribution over Q-values
                  "vmax": 25.,                      #maximum and minimum value of the distribution
                  "gamma": 0.99,                    #temporal discount factor
                  "update_horizon": 5,              #"forward-view" of TD error, i.e. how many time steps are
                                                        # optimized for at once
                  "min_replay_history": 500,        #minimum samples contained in replay buffer
                  "update_period": 4,               #
                  "target_update_period": 500,      #length of interval in between two updates of the target network
                  "epsilon_train": 0.0,
                  "epsilon_eval": 0.0,
                  "epsilon_decay_period": 1000,
                  "learning_rate": 0.000025,
                  "optimizer_epsilon": 0.00003125,
                  "tf_device": '/cpu:*'},
              "w_hint":1,
              "w_play":1,
              "w_disc":1,
              "dummy": 1,
              "pbt_mutables": ["dummy", "w_hint", "w_play", "w_disc"]  # list mutable parameters
            }

#### PBT runner
def main(unused_argv):
    """ Runs the self-play PBT training. """
    ### load or create population
    population, startstep = load_or_create_population(pbt_popsize, def_params)
    stats = [] #list of dicts that will be converted to pandas df
    ### pbt epoch loop
    for pbt_step in range(startstep, pbt_steps):
        print(f"{time.strftime('%a %d %b %H:%M:%S', time.gmtime())}: training pbt step {pbt_step}")
        stats_curr = ray.get([member.stepEval.remote() for member in population])
        #append global pbt step to dict and append to stats
        stats += stats_curr #append flat
        ### evolving members before next step
        perfs = [np.mean(stat_member["eval_episode_returns"]) for stat_member in stats_curr]
        print(perfs)
        die_idxs = list(
            np.argsort(perfs)[:pbt_discardN])  # idxs corresponding to worst 1-survRate members of population
        live_idxs = list(np.argsort(perfs)[pbt_discardN:])
        for idx, member in enumerate(population):
            if idx in die_idxs:
                # overwrite with better performing member in population
                exploit(member, [population[idx] for idx in live_idxs])
                member.mutate.remote() # pbt explore step

        ## checkpointing and bookkeeping
        if True:  # pbt_step != 0 and pbt_step % 10 == 0:
            save_population(population, stats)
    # writer.close()
    stats_df, stats = save_population(population, stats)
    plot_statistics(stats_df, def_params)
    # for member in population:
    #     member.agent._sess.close()
    return stats_df, stats



class Fake_flags(object):
    def __init__(self):
        self.base_dir = str(os.path.dirname(__file__)) + '/trained/PBTRainbow'
        self.checkpoint_dir = 'checkpoints'
        self.logging_dir = 'logs'
        self.game_type = 'Hanabi-Full'
FLAGS = Fake_flags()

stats_df, stats = main("hello")






