
import tensorflow as tf
import numpy as np
from version import *

class Module(object):
    def __init__(self,name,scope='',logging = False):
        self.name = name
        self.scope = scope

        self.logging = logging

    def __repr__(self):
        return self.scope + '/' + self.name


class GraphConvolutionLayer(object):
    '''
    one single Convulution layer
    '''

    def __init__(self, input_dim, output_dim, support, name,
                 scope='',keep_prob=1, act=tf.nn.relu, bias=True,
                 featureless=False, logging=True):
        '''
        :param featureless: If this is true, means don't do feature extraction in this layer
        :param logging: Whether to log the traing
        '''
        self.name = name
        self.scope = scope

        self.logging = logging

        self.keep_prob = keep_prob

        # one is for theta_0,the other is for theta_1
        self.support = support

        self.act = act
        self.featureless = featureless
        self.bias = bias

        self.vars = {}
        with tf.variable_scope(self.name + '/vars'):