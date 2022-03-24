
'''
Author: Luke
Date: 2019-9-4
'''


from Environment import *
from Model.GCN.Layer import Module
from Data import *

class Critic(Module):
    def __init__(self,args,input,env,scope='',logging = False):
        super(Critic,self).__init__(name='Critic',scope=scope,logging=logging)

        self.embedding_input = input
        self.batch_size = args['batch_size']

        with tf.variable_scope(self.name):
            with tf.variable_scope("Encoder/Initial_state"):
                # init states
                initial_state = tf.zeros([args['critic_rnn_layers'], 2, self.batch_size, args['critic_hidden_dim']],
                                         name='stacked_intial_state')
                l = tf.unstack(initial_state, axis=0,name='unstacked_state')
                rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
                                         for i in range(args['critic_rnn_layers'])])

                hy = tf.identity(rnn_tuple_state[0][1],name='hidden_state')

            with tf.variable_scope("Encoder/Process"):
                for i in range(args['critic_n_process_blocks']):
                    process = CriticAttentionLayer(args['critic_hidden_dim'], _name = "encoding_step_"+str(i),scope=self.scope + '/'+ self.name + '/Encoder/Process')