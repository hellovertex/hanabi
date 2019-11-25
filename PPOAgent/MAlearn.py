from PPOAgent.learn import *
from PPOAgent.Model import *
from PPOAgent.MAGame import *
import numpy as nps
    
def load_hanabi(ENV_CONFIG):
    # loads wrapped env
    return PyhanabiEnvWrapper(rl_env.make(**ENV_CONFIG))

def clip(rewards, min_val, max_val):
    return list(map(lambda x: max(min_val, min(x, max_val)), rewards))

def MAlearn(run_name, nupdates, k, ENV_CONFIG, MODEL_CONFIGS, wait_rewards = True,
            target_kl_init = 0.5, target_kl_goal = 0.01, kl_factor = 0.995,
            root_folder = './experiments/MA/', load = False, write_sum = True, train_schedule = 'par',
            updates_wait = 10):
    
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    load_env = lambda: load_hanabi(ENV_CONFIG)
    env = load_hanabi(ENV_CONFIG)
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    nactions = action_spec.maximum + 1 - action_spec.minimum
    nobs = obs_spec['state'].shape[0]
    nplayers = ENV_CONFIG['num_players']

    models = [Model(nactions, nobs, 1, sess = sess,
                     **MODEL_CONFIGS[i]) for i in range(nplayers)]
    
    savers = [tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = MC['scope'])) 
              for MC in MODEL_CONFIGS]
    
    game = MAGame(nplayers, models, load_env, wait_rewards = wait_rewards, 
                  train_schedule = train_schedule,  updates_wait = updates_wait)
    
    
   
    path = root_folder + ENV_CONFIG['environment_name'] + '-' + str(ENV_CONFIG['num_players'])
    if not os.path.isdir(path):
        os.makedirs(path)

    summary_writer = tf.summary.FileWriter(path + '/summary/' + run_name)
    
    loss_history, rewards_history, scores_history, lengths_history = defaultdict(list), [], [], []
    custom_rewards_history = {'hamming_distance_reward' : [],
                              'hint_last_copy_reward' : [],
                              'hint_playable_reward' : [],
                              'discard_extra_reward' : [],
                              'play_reward' : [],
                              'info_tokens_reward' : [],
                             }
    training_steps_history = []
    target_kl = target_kl_init
    if not os.path.isdir(path + '/models/' + run_name):
        os.makedirs(path + '/models/' + run_name)
    if load:
        for MC, saver in zip(MODEL_CONFIGS, savers):
            scope = MC['scope']
            ckpt = tf.train.get_checkpoint_state(path + '/models/' + run_name + '/%s/'% scope)
            saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    init_num_upd = sess.run(models[0].updates)
    if train_schedule == 'turn':
        init_num_upd *= 2
    print('Init update', init_num_upd)
    for nupd in range(0, nupdates):
        time_start = time.time()
        steps_start = game.total_steps
        
        score, rewards, lengths, custom_rewards = game.play_untill_train()
        for key in custom_rewards_history:
            custom_rewards_history[key].extend(custom_rewards[key])
        scores_history.append(np.mean(score))
        rewards_history.append(np.mean(rewards))
        lengths_history.append(np.mean(lengths))
        
        state_probs, policy_losses, value_losses, policy_entropyes, train_steps = game.train_model(k, target_kl)
        for pnum in policy_losses:
            loss_history[pnum].append((policy_losses[pnum], value_losses[pnum], policy_entropyes[pnum]))
            training_steps_history.append(train_steps[pnum])
            
        target_kl = max(target_kl_goal, kl_factor * target_kl)
        time_end = time.time()
        steps_end =  game.total_steps
        wasted_steps = steps_end - steps_start - models[0].nbatch
        wasted_percenteage = wasted_steps / (steps_end - steps_start)
        
        if nupd % 50 == 0:
            print('-----------------UPDATE%d----------------' % (nupd + init_num_upd))
            print('R:%.2f' % np.mean(rewards_history), 
                  'Score: %.2f' % np.mean(scores_history),
                  'Length: %d' % np.mean(lengths_history))
            print('%.1f steps/second' % ((steps_end-steps_start)/(time_end-time_start)) )
            print('%d steps in total' % steps_end)
            print('%.2f steps wasted' % wasted_percenteage)
            print('Updates/batch: %d' % int(np.mean(training_steps_history)))

            for MC, saver in zip(MODEL_CONFIGS, savers):
                scope = MC['scope']
                saver.save(sess, 
                           path+'/models/'+run_name+'/' + scope + '/model-' +str(init_num_upd+nupd)+'.cptk')
                
                
            print('-------------------------------------------')
            
            training_steps_history = []
                  
        if write_sum:
            if nupd % 10 == 0 and nupd > 0:
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', 
                                  simple_value = float(np.mean(rewards_history)))
                summary.value.add(tag='Perf/Score', 
                                  simple_value = float(np.mean(scores_history)))
                summary.value.add(tag='Perf/EpLength', 
                                  simple_value = float(np.mean(lengths_history)))
                summary.value.add(tag='Perf/Training steps per epoch', 
                                  simple_value = float(np.mean(training_steps_history)))
                for pnum in range(nplayers):
                    player_loss_history = np.array(loss_history[pnum])
                    if len(player_loss_history) == 0:
                        continue
                    summary.value.add(tag='Losses/Player%d/Policy Loss'% pnum, 
                                      simple_value = float(np.mean(player_loss_history[:, 0])))
                    summary.value.add(tag='Losses/Player%d/Value Loss'% pnum,
                                      simple_value = float(np.mean(player_loss_history[:, 1])))
                    summary.value.add(tag='Losses/Player%d/Entropy'% pnum, 
                                      simple_value = float(np.mean(player_loss_history[:, 2])))
                    '''summary.value.add(tag = 'Perf/Player%d/Actions'% pnum, 
                                      histogram = action_distribution_history[pnum])'''
                    
                summary_writer.add_summary(summary, nupd + init_num_upd)
                summary_writer.flush()
                loss_history = defaultdict(list)
                rewards_history, scores_history, lengths_history = [], [], []
                training_steps_history = []
    return game, model, sess
                  
                  