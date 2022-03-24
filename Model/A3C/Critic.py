
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
                    e, logit = process(hy, self.embedding_input, env)

                    prob = tf.nn.softmax(logit)
                    # hy : [batch_size x 1 x sourceL] * [batch_size  x sourceL x hidden_dim]  ->
                    # [batch_size x h_dim ]
                    hy = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), e), 1)

            with tf.variable_scope("Decoder"):
                self.reward_predict = tf.layers.dense( tf.layers.dense(hy, args['critic_hidden_dim'], tf.nn.relu, name='Full_Connect_L1'),
                                                            1, name='Full_Connect_L2')

            self.reward_predict = tf.squeeze(self.reward_predict,1, name= 'Reward_Predict')

        if self.logging == True:
            self._log()


    def _log(self):
        if self.scope == '':
            scope = self.name
        else:
            scope = self.scope + '/' + self.name
        for var in tf.trainable_variables(scope=scope):
            if self.scope == '':
                tf.summary.histogram(var.name, var)
            else:
                tf.summary.histogram(var.name[len(self.scope) + 1:], var)



class CriticAttentionLayer(object):
    """A generic attention module for the attention in vrp model"""

    def __init__(self, dim, _name,use_tanh=False, C=10,scope=''):
        self.scope = scope
        self.call_time = 0
        self.use_tanh = use_tanh
        self.name = _name

        with tf.variable_scope(self.name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v_vector', [1, dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v, 2, name='expand_v_vector')

            self.emb_d = tf.keras.layers.Conv1D(dim, 1, name= self.name + '/emb_d')
            self.project_d = tf.keras.layers.Conv1D(dim, 1, name= self.name + '/proj_d')

            self.project_query = tf.keras.layers.Dense(dim, name= self.name + '/proj_q')
            self.project_ref = tf.keras.layers.Conv1D(dim, 1, name= self.name + '/proj_ref')

        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args:
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder.
                [batch_size x max_time x dim]

            env: keeps demand ond load values and help decoding. Also it includes mask.
                env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any
                         positive number in this mask means that the node cannot be selected as next
                         decision point.
                env.demands: a list of demands which changes over time.

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # we need the first demand value for the critic
        self.call_time += 1
        demand = tf.identity(env.demand_trace[-1],name= self.name + '/demand')
        max_time = tf.identity(tf.shape(demand)[1],name= self.name + '/maxtime')
