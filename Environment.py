
'''
Author: Luke
Date: 2019-9-2
'''

import tensorflow as tf
import collections
import time
import numpy as np
import contextlib
import scipy.io as sio


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class Environment(object):
    def __init__(self, args, beam_width=1):
        print('Create environment...')
        start_time = time.time()
        self.args = args

        self.n_customers = args['n_customers']
        self.depot_id = self.n_customers
        self.n_nodes = self.n_customers + 1
        self.capacity = args['capacity']
        self.batch_size = args['batch_size']

        self.beam_width = beam_width
        self.batch_beam = self.batch_size * self.beam_width

        self.demand_trace = []
        self.load_trace = []
        self.mask_trace = []

        self.reward_trace = []
        self.prob_trace = []

        with tf.variable_scope('Environment'):
            with tf.variable_scope('Costumer_Info'):
                self.input_pnt = tf.placeholder(tf.float32, shape=[None, self.n_nodes, 2], name='Input_pnt')

                self.input_distance_matrix = tf.placeholder(tf.float32, shape=[None, self.n_nodes, self.n_nodes],
                                                            name='Input_distance_matrix')

                init_demand = tf.placeholder(tf.float32, shape=[None, self.n_nodes], name='Initial_Input_demand')

            with tf.variable_scope('Vehicle_Info'):
                init_load = tf.multiply(tf.ones([self.batch_beam]), self.capacity, name='Initial_Load')

            self.demand_trace.append(init_demand)

            with tf.variable_scope('Mask'):
                mask = tf.concat([tf.cast(tf.equal(self.demand_trace[-1], 0), tf.float32)[:, :-1],
                                  tf.ones([self.batch_beam, 1])], 1, name='Initial_mask')

            self.load_trace.append(init_load)
            self.mask_trace.append(mask)

        model_time = time.time() - start_time
        print(f'It took {model_time:.2f}s to build the environment.')


    def info_inspect(self, input_data, model, sess):
        '''
        Resets the environment. The environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.

        :param beam_width: width of the beam search, default = 1 is greedy search
        :return: a snapshot of the env
        '''
        # dimensions
        with tf.variable_scope('Environment/Info_inspect'):
            print("Info Statistic!")
            for i in range(self.args['actor_decode_len'] + 1):
                prob = self.prob_trace[i].eval(session=sess, feed_dict={