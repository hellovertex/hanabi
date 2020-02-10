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
FLAGS = flags.FLAGS

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
        self.parent_idx = []  # save parent idx before every pbt step
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
                                                     agent_type=params["agent_type"],
                                                     tf_session=session)  # TODO: pass config of agent here

        self.statistics = []  # save statistics after every pbt step here

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
            self.statistics.append(stats_curr) #TODO: nice logging and statistics taking
        return stats_curr["eval_episode_returns"][0]

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
            (self.params, self.parent_idx,self.statistics,self.pbt_step) = pickle.load(
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
        # self.id and self.ckpt_dir remains the same
        self.params = newMember.params  # copy parameters and append parent index to history
        # self.environment, self.logger and self.obs_stacker also remain the same
        # now copy tf variables from other agent, rest remains the same
        old_scope = "agent" + self.id
        new_scope = "agent" + newMember.id
        for var in tf.global_variables(scope=old_scope):
            value = self.agent._sess.run(var.name.replace(old_scope, new_scope)) #look up new value
            var.load(value.copy(), self.agent._sess)  # load it, sessions of old and new agent should be the same
        # self.pbt_step and self.checkpointer also remain

def exploit(member, good_population):
    """Overwrites hyperparams and parameters of agent in member with a randomly chosen member of good_population"""
    newMember = good_population[np.random.choice(range(len(good_population)))]
    member.pbt_copy(newMember)
    print(f"overwrote member {member.id} with member {newMember.id}")

def explore_template(mutables, mutprob, mutstren):
    """Decorates explore function with PBT experiments mutation hyperparameters
    Args:
        mutables: list of params.keys that are mutable
        mutprob: probability that mutation occurs
        mutstren: maximum strength of mutation relative to current value

    Returns: parameterized explore function

    """
    def explore(member):
        """Mutates parameters in-place according to PBT experiments mutation hyperparameters."""
        for param in mutables:
            if np.random.uniform(0,1) < mutprob:
                member.params[param] += member.params[param] * np.random.uniform(-1,1) * mutstren
                #TODO: apply parameters to agent if they were changed!!!
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
    else: #start from fresh
        #members are good to run
        startstep = 0
    return(population, startstep, session)

def save_population(population):
    for member in population:
        member.save_ckpt()

def create_and_test_population(unused_argv):
    """Some tests for PBT functionality.

    Doesn't test everything that happens internal to tensorflow, so not 100% guarantee that everything
    works fine under the hood unfortunately.
    """

    # ## record all the funny business that's going on to display graph in tensorboard later on
    # writer = tf.summary.FileWriter('test_session_graph', session.graph)

    ## training / evaluation
    try:
        population, startstep, session = load_or_create_population(3, def_params)
        [m0, m1, m2] = population  # for more convenient access
        m0.train()
        m0.eval()
        m0.stepEval()
        print("train/eval O.K.")
    except:
        session.close()
        raise Exception("problem with train/eval")

    ## save / load
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

    for _ in range(10): #turns out since this is the beginning of the training and we picked the first layer's
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

#out of service
# def fuck_up_2(unused_argv):
#     population, startstep, session = load_or_create_population(3, def_params)
#     [m0, m1, m2] = population  # for more convenient access
#
#     writer = tf.summary.FileWriter('test_session_graph', session.graph)
#     #pick some tensor in first member's subgraph
#     v0 = [v for v in tf.global_variables() if v.name == 'agent000/Online/fully_connected/weights:0'][0]
#     v1 = [v for v in tf.global_variables() if v.name == 'agent001/Online/fully_connected/weights:0'][0]
#     v2 = [v for v in tf.global_variables() if v.name == 'agent002/Online/fully_connected/weights:0'][0]
#
#     vals = []
#     save = lambda: vals.append([session.run(v1), session.run(v2)])
#     save() #0
#
#     #evaluate anything
#     session.run(v0)
#     session.run(v1)
#     session.run(v2)
#     save() #1
#
#     #copy m0 -> m1, m2
#     m1.pbt_copy(m0)
#     m2.pbt_copy(m0)
#     save() #2
#
#     #step m1
#     for _ in range(10):
#         m1.stepEval()
#     save()
#
#     #step m0
#     for _ in range(10):
#         m0.stepEval()
#     save() #3
#
#     #step m1
#     for _ in range(10):
#         m1.stepEval()
#     save() #4
#
#     #step m2
#     for _ in range(10):
#         m2.stepEval()
#     save() #5
#
#     session.close()
#     for i,elem in enumerate(vals):
#         print(f"after operation {i}:")
#         print(np.array_equal(elem[0],elem[1]))
#     writer.close()

#out of service
# def fuck_up(unused_argv):
#     population, startstep, session = load_or_create_population(3, def_params)
#     [m0, m1, m2] = population  # for more convenient access
#     m0.stepEval()
#
#     m1.pbt_copy(m0)
#     m2.pbt_copy(m0)
#
#     v0 = [v for v in tf.global_variables() if v.name == 'agent000/Online/fully_connected/weights:0'][0]
#     v1 = [v for v in tf.global_variables() if v.name == 'agent001/Online/fully_connected/weights:0'][0]
#     v2 = [v for v in tf.global_variables() if v.name == 'agent002/Online/fully_connected/weights:0'][0]
#
#
#     session.run(v1)
#     session.run(v2)
#     m1.stepEval()
#     val1_0 = session.run(v1)
#     val2_0 = session.run(v2)
#     print(np.array_equal(val1_0,val2_0))
#
#     val0_1 = session.run(v0)
#     val1_1 = session.run(v1)
#     val2_1 = session.run(v2)
#
#     m0.stepEval()
#
#     val1_2 = session.run(v1)
#     val2_2 = session.run(v2)
#
#     m1.stepEval()
#
#     val0_3 = session.run(v0)
#     val1_3 = session.run(v1)
#     val2_3 = session.run(v2)
#
#     m2.stepEval()
#
#     val1_4 = session.run(v1)
#     val2_4 = session.run(v2)
#
#     m0.stepEval()
#
#     val1_5 = session.run(v1)
#     val2_5 = session.run(v2)
#
#
#     print("before and after copying:")
#     print(np.array_equal(val1_0, val1_1))
#     print("before and after stepping m0")
#     print(np.array_equal(val1_1, val1_2))
#     print("are m1 and m2 the same? after stepping m1?")
#     print(np.array_equal(val1_3, val2_3))
#     print("and what about m1 == m0?")
#     print(np.array_equal(val1_3, val0_3))
#     print("what about stepping m2? m1 and m2 the same?")
#     print(np.array_equal(val1_4, val2_4))
#     print("now m0 step? m1 and m2 the same?")
#     print(np.array_equal(val1_5, val2_5))

#### PBT hyperparameters and default model params

pbt_steps = 50
pbt_popsize = 3
pbt_mutprob = 0.3  # probability for a parameter to mutate
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
              "RainbowAgent.update_horizon": 5,     #agent's params
              "RainbowAgent.num_atoms": 51,
              "RainbowAgent.min_replay_history": 500,  # agent steps
              "RainbowAgent.target_update_period": 500,  # agent steps
              "RainbowAgent.epsilon_train": 0.0,
              "RainbowAgent.epsilon_eval": 0.0,
              "RainbowAgent.epsilon_decay_period": 1000,  # agent steps
              "RainbowAgent.tf_device": '/cpu:*',  # '/gpu:0' use for non-GPU version
              "WrappedReplayMemory.replay_capacity": 50000,
              "dummy": 1}

# list mutable parameters
pbt_mutables = ["dummy"]
# generate parameterized explore fn
explore = explore_template(pbt_mutables, pbt_mutprob, pbt_mutstren)


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
            else:
                member.parent_idx.append(idx) #survived this step
            member.pbt_step += 1
    pass
    session.close()

if __name__ == '__main__':
    # flags.mark_flag_as_required('base_dir')
    app.run(create_and_test_population)
    # app.run(main)
        #todo: change verbosity of train steps
        #todo: logging properly
        #todo: passing variables properly
        #todo: parallelize code :'( -> really necessary though? let's check how tf arranges on the large machines
