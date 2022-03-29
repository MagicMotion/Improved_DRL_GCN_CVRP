
from Model.A3C.Actor import *
from Model.A3C.Critic import *
from Model.GCN.GCN import *
from Data import *
import json


class LinearEmbedding(Module):
    def __init__(self, args, scope='', logging=False):
        super(LinearEmbedding, self).__init__(name='LinearEmbediding', scope=scope, logging=logging)

        self.units = args['linear_embedding_num_units']
        if args['linear_embedding_use_tanh'] :
            self.layer = tf.layers.Dense(self.units, use_bias=args['linear_embedding_use_bias'])
        else :
            self.layer = tf.layers.Dense(self.units, activation=tf.nn.tanh, use_bias=args['linear_embedding_use_bias'])


        self.keep_prob = args['keep_prob']

    def __call__(self, inputs, *args, **kwargs):
        with tf.variable_scope(self.name):
            input = tf.cast(inputs['input_pnt'], dtype=tf.float32)

            input = tf.nn.dropout(input, keep_prob=self.keep_prob)
            with tf.variable_scope('Forward'):
                self.output = self.layer(input)

        # layer's variables will be created after invoked
        if self.logging:
            self._log_vars()

        return self.output

    def _log_vars(self):
        # tf.trainable_variables() use absolute path to locate variable, and yet tf.summary.histogram() use relative path
        # so I need to rename the scope to make sure the right relationship
        for var in tf.trainable_variables(scope=self.scope + '/' + self.name):
            tf.summary.histogram(var.name[len(self.scope) + 1:], var)


class Model(object):
    def __init__(self, args, Inputs, env, embedding_type='gcn',operation='train'):
        self.name = 'Model'

        #input include coordinates,demand,adjacent matrix
        self.inputs = Inputs

        with tf.variable_scope(self.name):
            if embedding_type == 'gcn':
                self.gcn = GCN(args, scope=self.name, logging=True, inputs=self.inputs)
                self.embedding = self.gcn.outputs
            elif embedding_type == 'linear_embedding':
                self.linear_layer = LinearEmbedding(args, scope=self.name, logging=True)
                self.embedding = self.linear_layer(self.inputs)

            if operation == 'train':
                print('Use greedy policy to train...')
                self.actor = Actor(args, input=self.embedding, env=env, scope=self.name,logging=True,mode='greedy')
            elif operation == 'infer':
                self.actor = Actor(args, input=self.embedding, env=env, scope=self.name,logging=True,mode='greedy')

            self.critic = Critic(args, self.embedding, env, scope=self.name,logging=True)

            self.outputs = self.actor.outputs

    def inference(self):
        pass


def record_args(args, file_name='args.json'):
    with open(file_name, 'w') as f:
        json.dump(args, f, sort_keys=True, indent=4, separators=(',', ': '))