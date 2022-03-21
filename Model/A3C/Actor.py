
'''
Author: Luke
Date: 2019-9-4
This module is the implmentation of the actor which makes the decision
'''

from Environment import *
from Model.GCN.Layer import Module
from Data import *


class Actor(Module):
    def __init__(self, args, input, env, scope='', logging=False,mode='greedy'):
        super(Actor, self).__init__(name='Actor', scope=scope, logging=logging)

        self.args = args

        self.embedding_input = tf.identity(input, name=self.name + '/Embedding_input')
        # declare variable in Actor
        with tf.variable_scope(self.name):
            self.decodeStep = RNNDecodeStep(self.args, scope=self.scope + '/' + self.name)

        # build computation data flow
        self.set_decision_mode(env,decode_type=mode)

    def set_decision_mode(self, env, decode_type="greedy"):
        '''
        set how to make decision
        :param env: The environment
        :param decode_type: ['greedy','beam_search','stochastic']
        :return: solution of the instance
        '''
        with tf.variable_scope(self.name + '/Forward'):
            batch_size = self.args['batch_size']

            if decode_type == 'greedy' or decode_type == 'stochastic':
                self.beam_width = 1
            elif decode_type == 'beam_search':
                self.beam_width = self.args['beam_width']

            BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size * self.beam_width), tf.int64), 1,