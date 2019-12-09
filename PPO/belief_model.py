import tensorflow as tf
from .lstm_util import multilayer_fc
import numpy as np
class BeliefModel():
    def __init__(self, nobs, nactions, scope, sess, path, hidden_units = 32):
        self.path = path + scope
        self.scope = scope
        self.sess = sess
        self.nactions = nactions
        with tf.variable_scope(scope, reuse = False):
            self.INP = tf.placeholder(dtype = 'float32', shape = [None, nobs])
            #self.LM = tf.placeholder(shape = [None, nactions], dtype = 'float32')
            self.A = tf.placeholder(shape = [None], dtype = 'int32', name = 'actions')
            self.A_OH = tf.one_hot(self.A, nactions, dtype = 'float32', name = 'actions_oh')
            if hidden_units is not None and hidden_units > 0 :
                h = multilayer_fc(self.INP, [hidden_units], scope = 'fc_net', layer_norm = False)
            else:
                h = self.INP
            
            self.output = multilayer_fc(h, [nactions], scope = 'output', layer_norm = False,
                                   activation = None,)
            self.output_probs = tf.nn.softmax(self.output)
            self.output_log_probs = tf.nn.log_softmax(self.output)
            
            self.sample_action = tf.squeeze(tf.multinomial(self.output, 1), axis = 1)
            self.sample_action_oh = tf.one_hot(self.sample_action, depth = nactions)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.A_OH, 
                                                                            logits = self.output_log_probs))
            params = tf.trainable_variables(scope)
            
            grads = tf.gradients(self.loss, params)
            grads = list(zip(grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate = 1e-3, )
            self._train = trainer.apply_gradients(grads)
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.sess.run(tf.variables_initializer(variables))
        self.saver = tf.train.Saver(var_list = variables)
    
    def train(self, nepochs, inp_train, actions_train):
        feed_dict = {self.INP : inp_train, self.A : actions_train}
        loss_hist = []
        for i in range(nepochs):
            loss_eval, _, output_probs_eval  = self.sess.run([self.loss, self._train, self.output_probs], 
                                                        feed_dict = feed_dict)
            loss_hist.append(loss_eval)
        return np.mean(loss_hist)
    
    def run_sample_actions(self, inp,):
        self.saver.save(self.sess, self.path + 'model/model.cptk')
        feed_dict = {self.INP : inp}
        actions, probs  = self.sess.run([self.sample_action, self.output_probs], feed_dict = feed_dict)
        return actions, probs

    def compute_action_probs(self, inp):
        # how: 'probs', 'hidden', 'weights'
        new_inp = [] # deleting observed cards
        for inp_env in inp:
            for i in range(10):
                for j in range(10):
                    hand_vec = np.zeros(20)
                    hand_vec[[i, 10 + j]] = 1
                    new_inp.append(np.concatenate([hand_vec, inp_env[20:]],0))
        new_inp = np.array(new_inp)
        #print('BM input', new_inp.shape)
        feed_dict = {self.INP : new_inp }
        probs = self.sess.run([self.output_probs], feed_dict = feed_dict)[0].reshape((inp.shape[0],
                                                                                      10, 10,-1))
        return probs

    def save_model(self):
        self.saver.save(self.sess, self.path + '/model.cptk')
        
    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.path)
        if ckpt is None:
            print('Could not load model "%s" at %s' % (self.scope, self.path))
        else:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)