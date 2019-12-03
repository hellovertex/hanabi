import numpy as np
import tensorflow as tf

def randomize_dict(d, min_val = 0.3, max_val = 2.5):
    for key in d:
        d[key] *= np.random.uniform(min_val, max_val)
    return d

def mutate_dict(d, mutation_fun, mutation_prob = 0.075):
    d_new = dict(d)
    for key in d_new:
        if np.random.uniform(0, 1) <= mutation_prob:
            mutation = mutation_fun()
            d_new[key] *= mutation
    return d_new


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

    
