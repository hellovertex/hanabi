from PPOAgent.learn import *
from PPOAgent.Model import *
from PPOAgent.MAGame import *
import numpy as np

def load_hanabi(ENV_CONFIG):
    # loads wrapped env
    return PyhanabiEnvWrapper(rl_env.make(**ENV_CONFIG))

def clip(rewards, min_val, max_val):
    return list(map(lambda x: max(min_val, min(x, max_val)), rewards))

def gen_run_name(model_type, MODEL_CONFIG, ENV_CONFIG, k, target_kl, clip_val):
    def add(name, term):
        return name + term + '_'
    name = model_type + '_'
    net = 'fc'
    for fc_size in MODEL_CONFIG['fc_input_layers']:
        net += str(fc_size) + '-'
    net = net[: -1]
    if MODEL_CONFIG['noisy_fc']:
        net += 'noisy'
        
    if len(MODEL_CONFIG['lstm_layers']):
        net_lstm = 'lstm'
        for lstm_size in MODEL_CONFIG['lstm_layers']:
            net_lstm += str(lstm_size) + '-'
        net_lstm = net_lstm[: -1]
        if MODEL_CONFIG['noisy_lstm']:
            net_lstm += 'noisy'
        net = add(net + '_', net_lstm) 
        
    net = add(net, MODEL_CONFIG['v_net'])
    name = add(name, net[:-1])
    name = add(name, 'lr%.4f' % MODEL_CONFIG['lr'])
    
    name = add(name, '%denvs_%dst' % (MODEL_CONFIG['nenvs'], MODEL_CONFIG['nsteps']))
    if model_type == 'PPO':
        name = add(name, 'k=%d' % k)
        name = add(name, 'KL=%.3f' % target_kl)
    name = add(name, 'clip=' + str(clip_val))
    name = add(name, 'gamma=%.2f' % MODEL_CONFIG['gamma'])
    name = add(name, 'vf=%.1f' % MODEL_CONFIG['vf_coef'])
    name = add(name, 'ent=%.2f' % MODEL_CONFIG['ent_coef'])
    return name
def MAlearn(run_name, nupdates, k, ENV_CONFIG, MODEL_CONFIG, clip_func, model_type = 'PPO',
            target_kl_init = 0.5, target_kl_goal = 0.01,
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

    models = [Model(nactions, nobs, 1, scope = 'P%d' % i, sess = sess,
                     **MODEL_CONFIG) for i in range(nplayers)]
    game = MAGame(nplayers, models, load_env, clip_func = clip_func, train_schedule = train_schedule, 
                  updates_wait = updates_wait)
    
    path = root_folder + ENV_CONFIG['environment_name'] + '-' + str(ENV_CONFIG['num_players'])
    if not os.path.isdir(path):
        os.makedirs(path)

    summary_writer = tf.summary.FileWriter(path + '/summary/' + run_name)
    saver = tf.train.Saver(max_to_keep=5)
    
    loss_history, rewards_history, clipped_rewards_history, lengths_history = defaultdict(list), [], [], []
    training_steps_history = defaultdict(list)
    target_kl = target_kl_init
    if not os.path.isdir(path + '/models/' + run_name):
        os.makedirs(path + '/models/' + run_name)
    if load:
        ckpt = tf.train.get_checkpoint_state(path + '/models/' + run_name)
        saver.restore(sess,ckpt. model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    init_num_upd = sess.run(models[0].updates)
    action_distribution_history = defaultdict(list)
    for nupd in range(0, nupdates):
        time_start = time.time()
        steps_start = game.total_steps
        
        clipped_rewards, rewards, lengths = game.play_untill_train()
        clipped_rewards_history.append(np.mean(clipped_rewards))
        rewards_history.append(np.mean(rewards))
        lengths_history.append(np.mean(lengths))
        
        policy_losses, value_losses, policy_entropyes, train_steps, act_dist = game.train_model(k,target_kl)
        for pnum in policy_losses:
            loss_history[pnum].append((policy_losses[pnum], value_losses[pnum], policy_entropyes[pnum]))
            training_steps_history[pnum].append(train_steps[pnum])
            action_distribution_history[pnum].extend(act_dist[pnum].tolist())
            
        target_kl = max(target_kl_goal, 0.9975 * target_kl)
        time_end = time.time()
        steps_end =  game.total_steps
        wasted_steps = steps_end - steps_start - models[0].nbatch
        wasted_percenteage = wasted_steps / (steps_end - steps_start)
        
        if nupd % 50 == 0 and nupd > updates_wait:
            print('-----------------UPDATE%d----------------' % (nupd + init_num_upd))
            print('R:%.2f' % np.mean(rewards_history), 
                  'R_clipped: %.2f' % np.mean(clipped_rewards_history),
                  'Length: %d' % np.mean(lengths_history))
            print('%.1f steps/second' % ((steps_end-steps_start)/(time_end-time_start)) )
            print('%d steps in total' % steps_end)
            print('%.2f steps wasted' % wasted_percenteage)
            for pnum in range(nplayers):
                print('Player %d: updates/batch: %d' % (pnum, int(np.mean(training_steps_history[pnum]))))
                dist = np.bincount(action_distribution_history[pnum])/len(action_distribution_history[pnum])
                
                print('Player %d: action distr is \n' % pnum, np.round(dist, 3))
                saver.save(sess, path+'/models/'+run_name+'/P%dmodel-'%pnum+str(init_num_upd+nupd)+'.cptk')
            print('-------------------------------------------')
            
            training_steps_history = defaultdict(list)
                  
        if write_sum:
            if nupd % 10 == 0 and nupd > updates_wait:
                
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', 
                                  simple_value = float(np.mean(rewards_history)))
                summary.value.add(tag='Perf/ClippedReward', 
                                  simple_value = float(np.mean(clipped_rewards_history)))
                summary.value.add(tag='Perf/EpLength', 
                                  simple_value = float(np.mean(lengths)))
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
                action_distribution_history = defaultdict(list)
                loss_history = defaultdict(list)
                rewards_history, clipped_rewards_history, lengths_history =  [], [], []
                  