import tensorflow as tf
import numpy as np


def get_confs():
    rewards_weights_base = {'play0': 1, 'play1': 3, 'play2': 9, 'play3': 27, 'play4': 81,
                            'baseline': 3,
                            'discard_last_copy': -100, 'discard_extra': 0.5,
                            'hint_last_copy': 0.2, 'hint_penalty': 0.1, 'hint_playable': 0.2,
                            'use_hamming': True, 'loose_life': -50}

    env_config = {'environment_name': 'Hanabi-Small', 'num_players': 2,
                  'use_custom_rewards': True, 'open_hands': False}

    model_config_base = {'scope': '', 'nenvs': 16,
                         'fc_input_layers': [128], 'noisy_fc': False, 'v_net': 'shared',
                         'gamma': 1, 'ent_coef': 0.01, 'vf_coef': 1, 'cliprange': 0.2,
                         'max_grad_norm': None, 'k': 24,
                         'lr': 1e-4, 'anneal_lr': False,
                         'normalize_advs': True, 'layer_norm': False,
                         }

    # these attributes of the model will evolove
    random_attributes = {'lr': lambda: np.random.uniform(1e-5, 1e-3),
                         'k': lambda: np.random.randint(2, 36),
                         'cliprange': lambda: np.random.uniform(0.05, 0.4),
                         'vf_coef': lambda: np.random.uniform(0.1, 1.7),
                         'ent_coef': lambda: np.random.uniform(0.001, 0.075),
                         'nsteps': lambda: np.random.randint(2, 40)}

    return rewards_weights_base, env_config, model_config_base, random_attributes


def constant(p):
    return 1


def linear(p):
    return 1 - p


def middle_drop(p):
    eps = 0.75
    if 1 - p < eps:
        return eps * 0.1
    return 1 - p


def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1 - p < eps:
        return eps
    return 1 - p


def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1 - p < eps1:
        if 1 - p < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - p


schedules = {
    'linear': linear,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v * self.schedule(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v * self.schedule(steps / self.nvalues)


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def fc(x, nh, scope, init_scale=1.0, init_bias=0.0, layer_norm=False):
    nin = x.get_shape()[1].value
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.random_uniform_initializer(-np.sqrt(3 / nin), np.sqrt(3 / nin)))
        h = tf.matmul(x, w) + b
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
    return h


def fc_noisy(x, nh, scope, init_scale=1.0, init_bias=0.0, layer_norm=False, noise_w=None, noise_b=None):
    nin = x.get_shape()[1].value

    with tf.variable_scope(scope):
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.random_uniform_initializer(-np.sqrt(3 / nin), np.sqrt(3 / nin)))
        if noise_w is None:
            noise_w = tf.placeholder(dtype=tf.float32, shape=[nin, nh])
        if noise_b is None:
            noise_b = tf.placeholder(dtype=tf.float32, shape=[nh])

        w_noise = tf.get_variable("w_noise", [nin, nh], initializer=tf.constant_initializer(0.017))
        b_noise = tf.get_variable("b_noise", [nh], initializer=tf.constant_initializer(0.017))
        w += tf.multiply(noise_w, w_noise)
        b += tf.multiply(noise_b, b_noise)
        h = tf.matmul(x, w) + b
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
    return h, noise_w, noise_b


def multilayer_fc(X, layers=[64, 64], scope='fc_net', activation=tf.tanh, layer_norm=False):
    num_layers = len(layers)
    with tf.variable_scope(scope):
        h = tf.layers.flatten(X)
        for i in range(num_layers):

            num_hidden = layers[i]
            h = fc(h, nh=num_hidden, scope='fc_%d' % i, init_scale=np.sqrt(2), layer_norm=layer_norm)
            if activation is not None:
                h = activation(h)
    return h


def multilayer_fc_noisy(X, layers=[64, 64], scope='fc_net', activation=tf.tanh, layer_norm=False,
                        noise=None):
    num_layers = len(layers)
    noise_list = []
    with tf.variable_scope(scope):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            num_hidden = layers[i]
            if noise is None:
                h, noise_w, noise_b = fc_noisy(h, nh=num_hidden, scope='fc_noisy_%d' % i, init_scale=np.sqrt(2),
                                               layer_norm=layer_norm)
            else:
                noise_w, noise_b = noise[2 * i: 2 * i + 2]
                h, noise_w, noise_b = fc_noisy(h, nh=num_hidden, scope='fc_noisy_%d' % i, init_scale=np.sqrt(2),
                                               layer_norm=layer_norm, noise_w=noise_w, noise_b=noise_b)

            noise_list.extend([noise_w, noise_b])
            if activation is not None:
                h = activation(h)
    return h, noise_list


def write_into_buffer(buffer, training_stats, policy_loss, value_loss, policy_entropy):
    # writes all data into buffer dict
    buffer['Perf/Score'].append(np.mean(training_stats['scores']))
    buffer['Perf/Reward'].append(np.mean(training_stats['rewards']))
    buffer['Perf/Length'].append(np.mean(training_stats['lengths']))
    buffer['Perf/Reward by "play"'].append(np.mean(training_stats['play_reward']))
    buffer['Perf/Reward by "discard"'].append(np.mean(training_stats['discard_reward']))
    buffer['Perf/Reward by "hint"'].append(np.mean(training_stats['hint_reward']))
    # buffer['Perf/Updates per batch'].append(k_trained)
    # buffer['Perf/Updates done'].append(updates)
    # buffer['Losses/KL loss'].append(np.mean(kl))
    buffer['Losses/Policy loss'].append(np.mean(policy_loss))
    buffer['Losses/Value loss'].append(np.mean(value_loss))
    buffer['Losses/Policy entropy'].append(np.mean(policy_entropy))


def train_model(game, model, player_nums='all'):
    (mb_obs, _, mb_actions, mb_probs, mb_logp, mb_legal_moves, mb_values, mb_returns,
     mb_dones, mb_masks, mb_states, mb_states_v, mb_noise) = game.collect_data(player_nums)

    policy_loss, value_loss, policy_entropy = model.train(mb_obs, mb_actions, mb_probs, mb_logp,
                                                          mb_legal_moves, mb_masks, mb_values, mb_returns,
                                                          mb_states, mb_states_v, mb_noise)

    return policy_loss, value_loss, policy_entropy

