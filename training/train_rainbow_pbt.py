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

from agents.rainbow_copy.third_party.dopamine import logger
import agents.run_experiment as run_experiment
import hanabi_learning_environment.rl_env as rl_env

import pickle

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
    def __init__(self, params, idx):
        """
        Args:
            params: all hyperparameters, including the immutable ones
            parent_idx: idx of the "parent"

        """
        self.ckpt_dir = os.path.join(FLAGS.base_dir, FLAGS.checkpoint_dir, str(idx).zfill(3))
        self.params = params #members parameters
        self.parent_idx = [idx] #save parent idx before every pbt step

        # Initialize the environment.
        self.environment = run_experiment.create_environment(game_type=params["game_type"],
                                                        num_players=params["num_players"])
        # Initialize the Logger object.
        self.logger = logger.Logger(os.path.join(FLAGS.base_dir, FLAGS.logging_dir, str(idx).zfill(3)))
        # Initialize the observation stacker.
        self.obs_stacker = run_experiment.create_obs_stacker(self.environment)
        # Initialize the agent.
        #TODO: when the graph is initialized inside the agent, are the variables properly scoped? what about the default_graph
        self.agent = run_experiment.create_agent(self.environment, self.obs_stacker, agent_type=params["agent_type"])  #TODO: pass config of agent here
        # Reload latest checkpoint, if available, and initialize Checkpointer object
        self.pbt_step, self.checkpointer = ( #pbt_step is current pbt_step
            run_experiment.initialize_checkpointing(self.agent,
                                                    self.logger,
                                                    self.ckpt_dir))

        self.statistics = [] #save statistics after every pbt step here

    def train(self):
        pass    #TODO: break up stepEval into individual functions
    def eval(self):
        pass
    def stepEval(self):
        """Training step followed by evaluation of learned model.

        Returns: average performance over evaluation epochs"""
        # training and evaluation step (run_one_iteration does both with params suitably set)
        for iteration in range(self.params["num_iterations"]):
            stats_curr = run_experiment.run_one_iteration(self.agent, self.environment, self.obs_stacker,
                                                          iteration,# + 1,  # to trick iteration % evaluate_every_n == 0
                                                          training_steps=self.params["training_steps"],
                                                          evaluate_every_n=1,#self.params["num_iterations"],
                                                          num_evaluation_games=self.params["num_evaluation_games"],
                                                          observers=None)
            self.statistics.append(stats_curr)
        return stats_curr["eval_episode_returns"][0]

    # def save_agent(self):
        # """Save checkpoint of agent into self.ckpt_dir so that it can be loaded and training resumed later on.
        #
        # save_ckpt uses this function to save the agent.
        # #TODO: check in what format agents need to have in order to be loaded into GUI, provide a second mode of
        # saving in this format here
        # """
        # agent_dictionary = self.agent.bundle_and_checkpoint(self.ckpt_dir, self.pbt_step)
        # self.checkpointer.save_checkpoint(self.pbt_step, agent_dictionary)
        # #agent can be loaded by initializing member in presence of ckpt_dir with corresponding name

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

    #environment has been recreated in init anyway, however #players and game_type cannot be changed anymore
        #todo: clean up old memberinfo checkpoints (dopamine checkpointing deletes all but 3 last ones i think)

    def load_ckpt(self):
        """Load the pickled information from checkpoint after "empty" member was initialized.

        The agent is already loaded during initalization provided the correct ckpt_dir. Same for pbt_step.
        """
        (self.params, self.parent_idx,self.statistics,self.pbt_step) = pickle.load(
                    open(os.path.join(self.ckpt_dir, f'memberinfo.{self.pbt_step-1}'), "rb"))
        return self.pbt_step

    def pbt_copy(self, newMember):
        """In the sense of PBT, copy model-related elements of newMember by loading from its checkpoint while retaining
        own statistics and environment.

        Copy agent and corresponding logger from newMember, copy params and update parent_idx list.
        parent_idx. ckpt_dir, environment, obs_stacker, pbt_step and statistics belong to member and thus need not be
        changed.
        """
        run_experiment.initialize_checkpointing(self.agent, self.logger,
                                                newMember.ckpt_dir) #overwrite agent and logger
        self.parent_idx.append(newMember.parent_idx[0])  # parent_idx at 0 is own index in population
        self.params = newMember.params #overwrite params
        self.checkpointer = newMember.checkpointer

def exploit(member, good_population):
    """Overwrites hyperparams and parameters of agent in member with a randomly chosen member of good_population"""
    newMember = good_population[np.random.choice(range(len(good_population)))]
    member.pbt_copy(newMember)

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
    #initializing the agents creates an empty ckpt directory, so we have to check before
    if os.path.isdir(os.path.join(FLAGS.base_dir, FLAGS.checkpoint_dir)):
        population = [Member(def_params.copy(), i) for i in range(pbt_popsize)]  # create dummy population and load
        loaded_steps = [member.load_ckpt() for member in population]
        if loaded_steps.count(loaded_steps[0]) != len(loaded_steps): #not all loaded pbt_steps are identical
            raise Exception("checkpointed models are at different pbt steps" + loaded_steps)
        startstep = loaded_steps[0]
    else:
        population = [Member(def_params.copy(), i) for i in range(pbt_popsize)]  # creat population
        startstep = 0
    return(population, startstep)

def save_population(population):
    for member in population:
        member.save_ckpt()

#### PBT runner
def main(unused_argv):
    """ Runs the self-play PBT training. """
    # pool = mp.Pool(mp.cpu_count())
    # set PBT hyperparameters
    pbt_steps = 30
    pbt_popsize = 3
    # pbt_popsize = mp.cpu_count() #of course integer multiples of number of possible workers makes sense,
    # if that number is too small to allow for sufficient variability in the population, perhaps use an integer multiple
    pbt_mutprob = 0.3  # probability for a parameter to mutate
    pbt_mutstren = 0.2  # size of interval around a parameter's value that new value is sampled from
    pbt_survivalrate = 0.75  # percentage of members of population to be mutated, rest will be replaced
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

    ### load or create population
    population, startstep = load_or_create_population(pbt_popsize, def_params)
    # ckpt, fpath = ckpt_exists()
    # if ckpt == -1:
    #     startstep = 0
    #     population = [Member(def_params.copy(), i) for i in range(pbt_popsize)]  # create population
    # else:
    #     population = load_population(fpath)
    #     startstep = ckpt + 1

    ### pbt epoch loop
    for pbt_step in range(startstep, pbt_steps):
        # ### parallel training
        # perfs = pool.map(lambda x: x.stepEval(), [member for member in population])  # synchronously parallel
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

        ### close sessions
        # pool.close()
        # for member in population:
        #     member.sess.close()

if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
