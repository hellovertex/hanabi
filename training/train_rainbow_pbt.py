"""The entry point for population based training (PBT) of a Rainbow agent on Hanabi.
Particularly, agents can be trained only in self-play, i.e. playing with exact copies of themselves.
However, agents inherit directly from hanabi_learning_environment.rl_env.Agent and use the canonical ObservationEncoder
python3 -um train_rainbow_pbt --base_dir="{base_dir}"
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np

import sys
print(sys.path)
from hanabi_learning_environment.agents.rainbow.third_party.dopamine import logger
import hanabi_learning_environment.agents.rainbow.run_experiment as run_experiment
import hanabi_learning_environment.rl_env as rl_env

import pickle
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import tensorflow as tf

#removed gin configuration
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

class Fake_flags(object):
    def __init__(self):
        self.base_dir = str(os.path.dirname(__file__)) + '/trained/PBTRainbow'
        self.checkpoint_dir = 'checkpoints'
        self.logging_dir = 'logs'
        self.game_type = 'Hanabi-Full'


#### PBT functionality
class Member(object):
    """Member of population, wraps a parameterized training routine"""
    def __init__(self, params, idx, session):
        """
        Args:
            params: all hyperparameters, including the immutable ones
            idx: idx of member in population, for convenience
            session: tf session that member connects with, stored in self.agent._sess
        """
        self.id = str(idx).zfill(3)  # identifier number string
        self.ckpt_dir = os.path.join(FLAGS.base_dir, FLAGS.checkpoint_dir, self.id)
        self.params = params  # members parameters
        self.parent_idx = None  # save parent idx before every pbt step
        # Initialize the environment.
        self.environment = run_experiment.create_environment(game_type=params["game_type"],
                                                             num_players=params["num_players"])
        # Initialize the Logger object.
        self.logger = logger.Logger(os.path.join(FLAGS.base_dir, FLAGS.logging_dir, self.id))
        # Initialize the observation stacker.
        self.obs_stacker = run_experiment.create_obs_stacker(self.environment)
        # Initialize the agent.
        with tf.variable_scope("agent" + self.id, reuse=tf.AUTO_REUSE) as scope:
            self.agent = run_experiment.create_agent(self.environment, self.obs_stacker,
                                                     agent_type=self.params["agent_type"],
                                                     tf_session=session,
                                                     config = self.params["rainbow_config"])
        self.train_step = 0
        self.statistics = []  # save statistics after every pbt step here (so when copying the agent from a
        # member the training progress will not be lost

    def init_sess_vars_ckpting(self, session):
        """Tensorflow boilerplate to make a freshly created member's agent usable.

        Connects member/agent to global session, initializes agent's own variables and initializes a checkpointer
        object.
        """
        #TODO: maybe include in init?
        # connect agent to session
        self.agent._sess = session
        # initialize all own variables
        own_vars = [var for var in tf.global_variables(scope="agent" + self.id)]
        session.run(tf.initializers.variables(own_vars))
        # Reload latest checkpoint, if available, and initialize Checkpointer object
        self.pbt_step, self.checkpointer = (  # pbt_step is current pbt_step
            run_experiment.initialize_checkpointing(self.agent,
                                                    self.logger,
                                                    self.ckpt_dir))

    def train(self):
        for iteration in range(self.params["num_iterations"]):
            stats_curr = run_experiment.run_one_iteration(self.agent, self.environment, self.obs_stacker,
                                                          iteration,
                                                          training_steps=self.params["training_steps"],
                                                          evaluate_every_n=None)
    def eval(self):
        #TODO: IMPLEMENT PROPERLY
        pass
        # stats_curr = run_experiment.run_one_iteration(self.agent, self.environment, self.obs_stacker,
        #                                               iteration=1,
        #                                               training_steps=0,
        #                                               evaluate_every_n=1,  # self.params["num_iterations"],
        #                                               num_evaluation_games=self.params["num_evaluation_games"])
        # return stats_curr["eval_episode_returns"][0]
    def stepEval(self):
        """For convenience: Training step followed by evaluation of learned model.
        A bit more convenient as checkpointing works cleaner for this at the moment, needs to be improved for
        individual train() and eval() functions.

        Returns: average performance over evaluation epochs"""
        # training and evaluation step (run_one_iteration does both with params suitably set)
        for iteration in range(self.params["num_iterations"]):
            stats_curr = run_experiment.run_one_iteration(self.agent, self.environment, self.obs_stacker,
                                                          iteration,# + 1,  # to trick iteration % evaluate_every_n == 0
                                                          training_steps=self.params["training_steps"],
                                                          evaluate_every_n=1,#self.params["num_iterations"],
                                                          # #TODO: is this really correct?
                                                          num_evaluation_games=self.params["num_evaluation_games"])
            self.train_step += self.params["training_steps"]
            self.stats_clean = {"train_return": stats_curr["average_return"][0],
                                "eval_episode_length": stats_curr["eval_episode_lengths"][0],
                                "eval_episode_return": stats_curr["eval_episode_returns"][0]}
            self.update_statistics(stats_curr)
        return stats_curr["eval_episode_returns"][0]

    def change_param(self, param, value):
        self.params[param] = value #overwrite value
        pass #TODO: IMPLEMENT

    # def change_param(self, param, value):
    #     """Applies changes in configuration of Rainbow agent.
    #
    #     Configuration parameters are contained in self.params.rainbow_config. Initialize a new Rainbow agent with
    #     these parameters and copy weights accordingly, similar to pbt_copy().
    #     """
    #     if param in ["num_atoms", "tf_device"]: #can't change the size of Q-distribution support or tf-device on the fly
    #         raise NotImplementedError("changes to the network design not implemented")
    #
    #     agent = self.agent
    #     self.params["rainbow_config"][param] = value #overwrite value
    #     #now reinitialize agent with upated config parameters.
    #     # this is the safest way to ensure consistency among all modules
    #     vars = {}
    #     agent_scope = "agent" + self.id
    #     #turns out it's enough to set reuse=tf.AUTO_REUSE so that "newly" created variables will just continue to exist
    #     # with
    #     # their current values as far as tensorflow is concerned. only the structure on top of them making a full
    #     # agent out of them is changed.
    #     with tf.variable_scope(agent_scope, reuse=True) as scope:
    #         self.agent = run_experiment.create_agent(self.environment, self.obs_stacker,
    #                                                  agent_type=self.params["agent_type"],
    #                                                  tf_session=agent._sess,
    #                                                  config=self.params["rainbow_config"])
    #         own_vars = [var for var in tf.global_variables(scope=agent_scope)]
    #         self.agent._sess.run(tf.initializers.variables(own_vars))
    #
    #     # with tf.variable_scope(agent_scope, reuse=True) as scope:
    #     #     for var in tf.global_variables(scope=agent_scope):
    #     #         print(var.name)
    #     #         value = self.agent._sess.run(var.name) #look up value
    #     #         vars[var.name] = value #save in dictionary for later
    #     #     self.agent = run_experiment.create_agent(self.environment, self.obs_stacker,
    #     #                                              agent_type=self.params["agent_type"],
    #     #                                              tf_session=agent._sess,
    #     #                                              config = self.params["rainbow_config"])
    #     #     #now load variables' values
    #     #     for var in tf.global_variables(scope=agent_scope):
    #     #         var.load(vars[var.name], self.agent._sess)

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
        pickle.dump((self.params,self.parent_idx,self.statistics,self.pbt_step),
                    open(os.path.join(self.ckpt_dir, f"memberinfo.{self.pbt_step}"), "wb"))
        #save agent using agent's checkpointer
        run_experiment.checkpoint_experiment(self.checkpointer, self.agent, self.logger,
                                  self.pbt_step, self.ckpt_dir, checkpoint_every_n=1)
        #checkpoint_experiment retains the previous 3 checkpoints and deletes all older ones (inside the definition
        # of DQNAgent tf.train.Saver(max_to_keep=3). Now do the same cleanup for memberinfo
        try:
            os.remove(os.path.join(self.ckpt_dir, f"memberinfo.{self.pbt_step-4}"))
        except:
            pass
        #todo: clean up old memberinfo checkpoints (dopamine checkpointing deletes all but 3 last ones
        #TODO: clean up old tf_ckpt meta, data and index, this makes the ckpt folder grow large over iterations

    def load_ckpt(self):
        """Load the pickled information from checkpoint after "empty" member was initialized.

        The agent is already loaded during initalization provided the correct ckpt_dir. Same for pbt_step.
        """
        try:
            (self.params, self.parent_idx,self.statistics, self.pbt_step) = pickle.load(
                open(os.path.join(self.ckpt_dir, f'memberinfo.{self.pbt_step-1}'), "rb")) #TODO check correct global logging
            return self.pbt_step
        except FileNotFoundError:
            print(f"No ckpt file found for Member {self.id}, starting from step zero")
            return 0

    def pbt_copy(self, newMember):
        """In the sense of PBT, copy model-related elements of newMember by loading from its checkpoint while retaining
        own statistics and environment.

        Copy agent and corresponding logger from newMember, copy params and update parent_idx list.
        parent_idx. ckpt_dir, environment, obs_stacker, pbt_step and statistics belong to member and thus need not be
        changed.
        """
        # self.id and self.ckpt_dir remain the same
        # copy parameters and append parent index to history
        for field_name in ["params", "train_step", "pbt_step"]:
            setattr(self,field_name,getattr(newMember,field_name))
        self.parent_idx = newMember.id
        # self.environment, self.logger and self.obs_stacker also remain the same
        # now copy tf variables from other agent, rest remains the same
        old_scope = "agent" + self.id
        new_scope = "agent" + newMember.id
        for var in tf.global_variables(scope=old_scope):
            value = self.agent._sess.run(var.name.replace(old_scope, new_scope)) #look up new value
            var.load(value.copy(), self.agent._sess)  # load it, sessions of old and new agent should be the same
        # self.pbt_step and self.checkpointer also remain
        # self.statistics also remain untouched, otherwise information would be lost when copying

    def update_statistics(self, stats_curr):
        """append current training statistics and values of mutables together with some overhead information to
        self.statistics
        """
        #append overhead business  to statistics so every entry in statistics is completely interpretable by itself
        stats_curr["id"] = self.id
        stats_curr["parent_idx"] = self.parent_idx
        stats_curr["pbt_step"] = self.pbt_step
        stats_curr["train_step"] = self.train_step
        #look up all current values of mutable variables
        current_mutable_vals = dict((par_name, self.params[par_name]) for par_name in self.params["pbt_mutables"])
        #append everything together as one dictionary to self.statistics
        self.statistics.append({**stats_curr, **current_mutable_vals})

def exploit(member, good_population):
    """Overwrites hyperparams and parameters of agent in member with a randomly chosen member of good_population"""
    newMember = good_population[np.random.choice(range(len(good_population)))]
    print(f"overwriting member {member.id} with member {newMember.id}")
    member.pbt_copy(newMember)

def explore_template(mutprob, mutstren):
    """Decorates explore function with PBT experiments mutation hyperparameters
    Args:
        mutables: list of params.keys that are mutable
        mutprob: probability that mutation occurs
        mutstren: maximum strength of mutation relative to current value

    Returns: parameterized explore function

    """
    def explore(member):
        """Mutates parameters in-place according to PBT experiments mutation hyperparameters."""
        for param in member.params["pbt_mutables"]:
            if np.random.uniform(0,1) < mutprob:
                print(f"mutating {param} of member {member.id}")
                # value = member.params["rainbow_config"][param] #modifies in-place but we are going to overwrite it
                value = member.params[param]
                value += value * np.random.uniform(-1,1) * mutstren
                if param=="gamma":
                    value = np.clip(value, 0,1)
                member.change_param(param, value)
    return explore

#TODO: so far logging occurs inside an agent, so if say member 1 is replaced by member 2, all of its own logs will
# be lost -> implement pbt's own loggings to keep track of pbt-related stuff
#TODO: also save all pbt parameters etc. into same place so that the entire experiment is properly documented

#### checkpointing
def load_or_create_population(pbt_popsize, def_params):
    #initializing the agents creates an empty ckpt directory, so we have to check before whether checkpoint dir exists
    ckpt_dir_exists = os.path.isdir(os.path.join(FLAGS.base_dir, FLAGS.checkpoint_dir))

    # set up tensorflow session and graph
    tf.reset_default_graph()
    session = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))  # more tensorflow boilerplate

    #initialize members in population, init_sess_cars_ckpting initializes all tensorflow-related stuff
    population = [Member(def_params.copy(), i, 'empty') for i in range(pbt_popsize)]
    for member in population:
        member.init_sess_vars_ckpting(session)

    if ckpt_dir_exists: #load checkpoint
        loaded_steps = [member.load_ckpt() for member in population]
        if loaded_steps.count(loaded_steps[0]) != len(loaded_steps): #not all loaded pbt_steps are identical
            raise Exception("checkpointed models are at different pbt steps" + loaded_steps)
        startstep = loaded_steps[0]
        for member in population:
            member.pbt_step = startstep
    else: #start from fresh
        #members are good to run
        startstep = 0
    return(population, startstep, session)

def save_population(population):
    """Saves all members by calling their checkpointers and also dumps pbt statistics as df to disk.

    Returns: statistics of pbt experiment as df
    """
    for member in population:
        print(f"saving member {member.id}")
        member.save_ckpt()
    stats_df = dump_statistics(population)
    return stats_df

def dump_statistics(population):
    """Extract statistics from all members and save as dataframe"""
    # for each member for each iteration create a list of entries
    all_stats = []
    for member in population:
        all_stats += member.statistics #statistics should contain already all necessary information to interpret
        # entries outside of member
    # #clean-up
    # for entry in all_stats:
    #     del entry["train_episode_lengths"]
    #     del entry["train_episode_returns"]
    #     entry["average_return"] = entry["average_return"][0]
    #     entry["eval_episode_lengths"] = entry["eval_episode_lengths"][0]
    #     entry["eval_episode_returns"] = entry["eval_episode_returns"][0]
    stats_df = pd.DataFrame(all_stats) #make df from list of dicts
    try:
        pickle.dump((stats_df, def_params), open(os.path.join(FLAGS.base_dir,"pbt_stats.pickle"), 'wb'))
    except Exception as e:
        print("writing statistics failed:")
        print(e)
    return stats_df

def plot_statistics(stats, def_params=None):
    """Convenience function to plot statistics of PBT run

    Args:
        stats_ref: either pandas DataFrame or filepath to pickle file
    """
    if os.path.isfile(stats):
        (stats, def_params) = pickle.load(open(stats, 'rb'))
    if not type(stats) == pd.DataFrame:
        raise TypeError("Statistics are no Pandas DataFrame")
    if def_params == None:
        raise Exception("Default parameters missing")

    #stats now is assumed to be a correctly formatted DataFrame
    #get ids and corresponding colors globally to keep colors consistent across plots
    ids = list(set(stats.id))
    n_membs = len(ids)
    # colors = cm.get_cmap("gist_rainbow", n_membs)
    marker_style = dict(linestyle=':', marker='o',
                        markersize=10, markerfacecoloralt='white')

    #evaluation accuracy of all models
    fig, ax = plt.subplots()
    ax.set_title("Members' eval. accuracy")
    for i, id in enumerate(ids): #iterate over members
        id_df = stats[stats.id==id] #pick subset of data
        ax.plot(id_df.train_step, id_df.eval_episode_returns, color=f'C{i}', label='member '+id)
    ax.legend()
    ax.set_ylabel(f'Accuracy averaged over {def_params["num_evaluation_games"]} evaluation epochs')
    ax.set_xlabel("Train Step")

    #values of mutables over time
    fig, ax = plt.subplots()
    ax.set_title("Members' mutable variable values")
    for i, id in enumerate(ids): #iterate over members
        id_df = stats[stats.id==id] #pick subset of data
        for mutable, fillstyle in zip(def_params["pbt_mutables"],Line2D.fillStyles):
            ax.plot(id_df.train_step, id_df[mutable], color=f'C{i}', label=f'member {id}: {mutable}',
                    fill_style = fillstyle, **marker_style)
    ax.legend()
    ax.set_ylabel("Value of mutable variables")
    ax.set_xlabel("Train Step")

    #winning model average return, eval episode lengths and eval episode returns
    winning_model_id = stats.iloc[stats.eval_episode_returns.idxmax()].id
    id_df = stats[stats.id == winning_model_id]  # pick subset of data
    measures = ["eval_episode_lengths", "eval_episode_returns"]
    fig, ax = plt.subplots()
    ax.set_title("Winning Model")
    for measure, fillstyle in zip(measures,Line2D.fillStyles):
        ax.plot(id_df.train_step, id_df[measure], color=f'C{ids.index(winning_model_id)}', label=f'member '
                f'{winning_model_id}: {measure}', fillstyle = fillstyle, **marker_style)
    ax.legend()
    ax.set_ylabel("Performance measure")
    ax.set_xlabel("Train Step")


# def fuck_up(unused_argv):
#     population, startstep, session = load_or_create_population(3, def_params)
#     [m0, m1, m2] = population  # for more convenient access
#     v1 = [v for v in tf.global_variables() if v.name == 'agent001/Online/fully_connected/weights:0'][0]
#     # writer = tf.summary.FileWriter('test_reload_agent_graph', session.graph) #so graph can be visualized
#     m0.stepEval()
#     m1.pbt_copy(m0)
#     val0 = session.run(v1)
#     print([v.name for v in tf.global_variables() if v.name.startswith('agent001')])
#
#     for par_name, par_val in {"vmax": 24.,  # try changing out all parameters
#                               "gamma": 0.95,  # temporal discount factor
#                               "update_horizon": 4,
#                               "min_replay_history": 400,
#                               "update_period": 5,
#                               "target_update_period": 400,
#                               "epsilon_train": 0.1,
#                               "epsilon_eval": 0.1,
#                               "epsilon_decay_period": 5000,
#                               "learning_rate": 0.00025,
#                               "optimizer_epsilon": 0.0003125}.items():
#         m1.change_param(par_name, par_val)
#     print("PARTYYYY")
#     print([v.name for v in tf.global_variables() if v.name.startswith('agent001')])
#     m1.stepEval()
#     val1 = session.run(v1)
#     if np.array_equal(val0, val1):
#         print("weights unchanged after changing hyperparam")
#     # okay so here we don#t really check all parameters because sometimes they are really stored far away,
#     # the point is that internally to the agent correct classes (like the replay memory are
#     if m1.agent.gamma == 0.95 and m0.agent.update_horizon == 4 and m0.agent._replay.memory._update_horizon == 4:
#         print("structure modified correctly")
#     m1.stepEval()
#     m1.save_ckpt()

def create_and_test_population(unused_argv):
    """Some tests for PBT functionality.

    Doesn't test everything that happens internal to tensorflow, so not 100% guarantee that everything
    works fine under the hood unfortunately, but if this goes through well it's a pretty safe bet.
    """

    # ## record all the funny business that's going on to display graph in tensorboard later on
    # writer = tf.summary.FileWriter('test_session_graph', session.graph)

    ## training / evaluation
    try:
        population, startstep, session = load_or_create_population(3, def_params)
        [m0, m1, m2] = population  # for more convenient access
        # m0.train()
        # m0.eval()
        m0.stepEval()
        print("train/eval O.K.")
    except:
        session.close()
        raise Exception("problem with train/eval")

    # save / load
    try:
        # overwrite base_dir so it can be deleted conveniently later
        FLAGS.base_dir = os.path.join(FLAGS.base_dir, "test")
        m0.save_ckpt()
        session.close()
        print("saving O.K.")
        #restart session entirely: creating new population resets graph
        population, startstep, session = load_or_create_population(3, def_params)
        [m0, m1, m2] = population #for more convenient access
        #now try to load
        m0.load_ckpt()
        m0.stepEval() #alive and kicking?
        print("loading O.K.")
    except:
        session.close()
        raise Exception("problem with saving/loading")
    finally:
        shutil.rmtree(FLAGS.base_dir) #delete directory

    ### modify parameters
    try:
        v0 = [v for v in tf.global_variables() if v.name == 'agent000/Online/fully_connected/weights:0'][0]
        # writer = tf.summary.FileWriter('test_reload_agent_graph', session.graph) #so graph can be visualized
        m0.stepEval()
        val0 = session.run(v0)
        for par_name, par_val in {"vmax": 24., #try changing out all parameters
                              "gamma": 0.95,                    #temporal discount factor
                              "update_horizon": 4,
                              "min_replay_history": 400,
                              "update_period": 5,
                              "target_update_period": 400,
                              "epsilon_train": 0.1,
                              "epsilon_eval": 0.1,
                              "epsilon_decay_period": 5000,
                              "learning_rate": 0.00025,
                              "optimizer_epsilon": 0.0003125}.items():
            m0.change_param(par_name, par_val)
        m0.stepEval()
        val1 = session.run(v0)
        if np.array_equal(val0,val1):
            print("weights unchanged after changing hyperparam")
        #okay so here we don#t really check all parameters because sometimes they are really stored far away,
        # the point is that internally to the agent correct classes (like the replay memory are
        if m0.agent.gamma == 0.95 and m0.agent.update_horizon == 4 and m0.agent._replay.memory._update_horizon == 4:
            print("structure modified correctly")
    except:
        raise Exception("Problem with modifying parameters of agent")
    finally:
        pass
        # writer.close()

    ## copy other member, check whether variables are initialized correctly and are independent from now on
    session.close()

    population, startstep, session = load_or_create_population(3, def_params)
    writer = tf.summary.FileWriter('test_session_graph', session.graph)

    [m0, m1, m2] = population  # for more convenient access
    #pick some tensor in first member's subgraph
    v0 = [v for v in tf.global_variables() if v.name == 'agent000/Online/fully_connected/weights:0'][0]
    #save values for later
    val0_0 = session.run(v0)
    #copy m0 -> m1, m2
    m1.pbt_copy(m0)
    m2.pbt_copy(m0)
    val0_1 = session.run(v0)
    #pick corresponding tensors in second and third members' subgraphs
    v1 = [v for v in tf.global_variables() if v.name == 'agent001/Online/fully_connected/weights:0'][0]
    v2 = [v for v in tf.global_variables() if v.name == 'agent002/Online/fully_connected/weights:0'][0]
    val1_0 = session.run(v1)
    val2_0 = session.run(v2)

    #the values should be the same now
    if np.array_equal(val0_0,val1_0) and np.array_equal(val0_0,val2_0):
        print("copying member: tf Variables are copied correctly")
    else:
        print("member 0 first layer weights:")
        print(val0_0)
        print("member 1 first layer weights:")
        print(val1_0)
        raise Exception("copying member failed: tf Variables not copied copied correctly")
    #now train and see whether variables are independent IN BOTH DIRECTIONS

    for _ in range(5): #turns out since this is the beginning of the training and we picked the first layer's
        # weights that sometimes the first stepEval doesn't change these weights _at all_ -> run a couple of iterations
        m1.stepEval() #changes variable in member 1
    val0_1 = session.run(v0) #this should not have changed first member's variables
    if not np.array_equal(val0_1,val0_0):
        raise Exception("copying member failed: target member changes source member")

    for _ in range(10):
        m0.stepEval()
    val2_1 = session.run(v2) #this should not have changed second member's variables
    if not np.array_equal(val2_1,val2_0):
        raise Exception("copying member failed: source member changes target member")

    for _ in range(10):
        m2.stepEval()
    val1_2 = session.run(v1)
    val2_2 = session.run(v2)
    if np.array_equal(val1_2,val2_2):
        raise Exception("copying member failed: 2 target members not independent")
    print("copying members O.K.: tf Variables are correctly copied and independent")

    ## goodbye
    writer.close()
    #graph can be displayed with this terminal command: tensorboard --logdir="./test_session_graph"
    session.close()

#### PBT hyperparameters and default model params
pbt_steps = 20
pbt_popsize = 3
pbt_mutprob = 1  # probability for a parameter to mutate
pbt_mutstren = 0.2  # size of interval around a parameter's value that new value is sampled from
pbt_survivalrate = 0.5  # percentage of members of population to be mutated, rest will be replaced
pbt_discardN = int(pbt_popsize * (1 - pbt_survivalrate))  # for convenience

### define default config for currently trained agent and decide which parameters can be mutated
# so far using rl_env.make encapsulation of config when creating the environment
def_params = {"game_type": 'Hanabi-Small',  # environment parameters
              "agent_type": 'Rainbow',
              "num_players": 2,
              "training_steps": 100,  # schedule for training and eval step for particular set of hyperparameters
              "num_iterations": 3,
              "num_evaluation_games": 20,
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
              "dummy": 1,
              "pbt_mutables": ["dummy"]  # list mutable parameters
            } #todo: decide whether to also change replay memories params

# generate parameterized explore fn
explore = explore_template(pbt_mutprob, pbt_mutstren)

#### PBT runner
def main(unused_argv):
    """ Runs the self-play PBT training. """
    ### load or create population
    population, startstep, session = load_or_create_population(pbt_popsize, def_params)

    ### pbt epoch loop
    for pbt_step in range(startstep, pbt_steps):
        #TODO: implement parallel training or train members sequentially and let tensorflow organize to use
        # resources optimally?
        perfs = [member.stepEval() for member in population]
        ### checkpointing and bookkeeping
        save_population(population)
        ### evolving members before next step
        die_idxs = list(np.argsort(perfs)[:pbt_discardN]) #idxs corresponding to worst 1-survRate members of population
        live_idxs = list(np.argsort(perfs)[pbt_discardN:])
        for idx, member in enumerate(population):
            if idx in die_idxs:
                exploit(member, [population[idx] for idx in live_idxs]) #overwrite with better performing member in
                # population
                explore(member)
            else: #else just note that member survived this step and is its own parent
                member.parent_idx = idx #survived this step
            #count step and save current values of mutables
            member.pbt_step += 1
    ### save and exit
    save_population(population)
    session.close()


FLAGS = Fake_flags()
population, startstep, session = load_or_create_population(pbt_popsize, def_params)
### pbt epoch loop
for pbt_step in range(startstep, pbt_steps):
    # TODO: implement parallel training or train members sequentially and let tensorflow organize to use
    # resources optimally?
    perfs = [member.stepEval() for member in population]
    ### checkpointing and bookkeeping
    save_population(population)
    ### evolving members before next step
    die_idxs = list(np.argsort(perfs)[:pbt_discardN])  # idxs corresponding to worst 1-survRate members of population
    live_idxs = list(np.argsort(perfs)[pbt_discardN:])
    for idx, member in enumerate(population):
        if idx in die_idxs:
            exploit(member, [population[idx] for idx in live_idxs])  # overwrite with better performing member in
            # population
            explore(member)
        else: #else just note that member survived this step and is its own parent
            member.parent_idx = idx  # survived this step

    for member in population: # count step and save current values of mutables
        member.pbt_step += 1
### save and exit
stats_df = save_population(population)
plot_statistics(stats_df, def_params)
session.close()


# if __name__ == '__main__':
#     # flags.mark_flag_as_required('base_dir')
#     # app.run(create_and_test_population)
#     # app.run(fuck_up)
#     app.run(main)
#
#     #todo: logging properly, plotting progress
#     #todo: run "dry", estimate time
#         #todo: parallelize code :'( -> really necessary though? let's check how tf arranges on the large machines
#     #todo: go through params to see whether they should be changed in the first place, write down
#         #todo: change only optimizer?
#         #todo: what about just no Adam optimizer?
#     #todo: custom rewards
#     #todo: why does graph still grow?
