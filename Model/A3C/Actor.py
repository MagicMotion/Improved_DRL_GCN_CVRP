
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
                                           name='batch_sequence')

            # results of the decisions
            idxs = []
            actions_tmp = []
            logprobs = []
            probs = []

            # start from depot
            #all initial state of LSTM is set to zero
            with tf.variable_scope('LSTM/LSTM_decode_step/State/Initial_State'):
                # decoder_state
                initial_state = tf.zeros(
                    [self.args['actor_rnn_layer_num'], 2, batch_size * self.beam_width, self.args['actor_hidden_dim']],
                    name='stacked_intial_state')

                l = tf.unstack(initial_state, axis=0, name='unstacked_state')
                decoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
                                       for i in range(self.args['actor_rnn_layer_num'])])

            # start from depot in VRP
            # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
            decoder_input = tf.tile(tf.expand_dims(self.embedding_input[:, env.n_nodes - 1], 1),
                                    [self.beam_width, 1, 1], name='Decoder_input')

            idx = tf.multiply(tf.ones([batch_size, 1], tf.int64), env.depot_id)
            idxs.append(idx)

            # Sequence = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int64), 1)

            # action = tf.gather_nd(env.input_pnt, tf.concat([Sequence, idx], 1))
            sequence = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int64), 1)
            action = tf.gather_nd(env.input_pnt, tf.concat([sequence, idx], 1), name='start_position')

            actions_tmp.append(action)

            original_demand = tf.divide(tf.reduce_sum(env.demand_trace[0], axis=1), 2)
            # demand_panalty = tf.divide(tf.reduce_sum(env.demand_trace[0], axis=1),original_demand)

            # demand_penalty = tf.zeros_like(original_demand)
            # env.record_reward(tf.add(env.get_reward(actions_tmp), demand_penalty, name='reward'))

            # env.record_reward(env.get_reward(actions_tmp))
            env.record_reward(env.get_reward(actions_tmp[1:]))


            prob = tf.concat(
                [tf.zeros([batch_size, env.n_nodes - 1], tf.float32), tf.ones([batch_size, 1], tf.float32)], axis=1)
            probs.append(prob)
            env.record_prob(prob)

            logprob = tf.zeros([batch_size], tf.float32)
            logprobs.append(logprob)

            # decoding loop
            context = tf.tile(self.embedding_input, [self.beam_width, 1, 1], name='Context')
            for i in range(self.args['actor_decode_len']):
                # decoder input is the last chosen position's embedding input
                # logit is the masked output of decision,prob is masked decision prob,
                logit, prob, logprob, decoder_state = self.decodeStep.step(decoder_input,
                                                                           context,
                                                                           env,
                                                                           decoder_state)
                # idx: [batch_size*beam_width x 1]
                beam_parent = None
                if decode_type == 'greedy':
                    idx = tf.expand_dims(tf.argmax(prob, 1), 1)
                elif decode_type == 'stochastic':
                    # select stochastic actions. idx has shape [batch_size x 1]
                    # tf.multinomial sometimes gives numerical errors, so we use our multinomial :(
                    def my_multinomial():
                        prob_idx = tf.stop_gradient(prob)
                        prob_idx_cum = tf.cumsum(prob_idx, 1)
                        rand_uni = tf.tile(tf.random_uniform([batch_size, 1]), [1, env.n_nodes])
                        # sorted_ind : [[0,1,2,3..],[0,1,2,3..] , ]
                        sorted_ind = tf.cast(tf.tile(tf.expand_dims(tf.range(env.n_nodes), 0), [batch_size, 1]),
                                             tf.int64)
                        tmp = tf.multiply(tf.cast(tf.greater(prob_idx_cum, rand_uni), tf.int64), sorted_ind) + \
                              10000 * tf.cast(tf.greater_equal(rand_uni, prob_idx_cum), tf.int64)

                        idx = tf.expand_dims(tf.argmin(tmp, 1), 1)
                        return tmp, idx

                    tmp, idx = my_multinomial()
                    # check validity of tmp -> True or False -- True mean take a new sample
                    tmp_check = tf.cast(
                        tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tmp, 1), (10000 * env.n_nodes) - 1),
                                              tf.int32)), tf.bool)
                    tmp, idx = tf.cond(tmp_check, my_multinomial, lambda: (tmp, idx))
                elif decode_type == 'beam_search':
                    if i == 0:
                        # BatchBeamSeq: [batch_size*beam_width x 1]
                        # [0,1,2,3,...,127,0,1,...],
                        batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(batch_size), tf.int64),
                                                              [self.beam_width]), 1)
                        beam_path = []
                        log_beam_probs = []
                        # in the initial decoder step, we want to choose beam_width different branches
                        # log_beam_prob: [batch_size, sourceL]
                        log_beam_prob = tf.log(tf.split(prob, num_or_size_splits=self.beam_width, axis=0)[0])

                    elif i > 0:
                        log_beam_prob = tf.log(prob) + log_beam_probs[-1]
                        # log_beam_prob:[batch_size, beam_width*sourceL]
                        log_beam_prob = tf.concat(tf.split(log_beam_prob, num_or_size_splits=self.beam_width, axis=0),
                                                  1)

                    # topk_prob_val,topk_logprob_ind: [batch_size, beam_width]
                    topk_logprob_val, topk_logprob_ind = tf.nn.top_k(log_beam_prob, self.beam_width)

                    # topk_logprob_val , topk_logprob_ind: [batch_size*beam_width x 1]
                    topk_logprob_val = tf.transpose(tf.reshape(
                        tf.transpose(topk_logprob_val), [1, -1]))

                    topk_logprob_ind = tf.transpose(tf.reshape(
                        tf.transpose(topk_logprob_ind), [1, -1]))

                    # idx,beam_parent: [batch_size*beam_width x 1]
                    idx = tf.cast(topk_logprob_ind % env.n_nodes, tf.int64)  # Which city in route.
                    beam_parent = tf.cast(topk_logprob_ind // env.n_nodes, tf.int64)  # Which hypothesis it came from.

                    # batchedBeamIdx:[batch_size*beam_width]
                    batchedBeamIdx = batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_parent
                    prob = tf.gather_nd(prob, batchedBeamIdx)

                    beam_path.append(beam_parent)
                    log_beam_probs.append(topk_logprob_val)

                batched_idx = tf.concat([BatchSequence, idx], 1)

                decoder_input = tf.expand_dims(tf.gather_nd(
                    tf.tile(self.embedding_input, [self.beam_width, 1, 1]), batched_idx), 1)

                logprob = tf.log(tf.gather_nd(prob, batched_idx))
                probs.append(prob)
                idxs.append(idx)
                logprobs.append(logprob)

                action = tf.gather_nd(tf.tile(env.input_pnt, [self.beam_width, 1, 1]), batched_idx)
                actions_tmp.append(action)

                env.record_prob(prob)
                left_demand = env.response(idx, beam_parent)

                # demand_penalty = tf.divide(tf.add(left_demand, demand_penalty), original_demand, name='demand_penalty')
                # env.record_reward(tf.add(env.get_reward(actions_tmp), demand_penalty, name='reward'))

                # env.record_reward(env.get_reward(actions_tmp))
                env.record_reward(env.get_reward(actions_tmp[1:]))

            if decode_type == 'beam_search':
                # find paths of the beam search
                tmplst = []
                tmpind = [BatchSequence]
                for k in reversed(range(len(actions_tmp))):
                    tmplst = [tf.gather_nd(actions_tmp[k], tmpind[-1])] + tmplst
                    tmpind += [tf.gather_nd(
                        (batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_path[k]), tmpind[-1])]
                actions = tmplst
            else:
                actions = actions_tmp

            env.trace_actor_act(idxs, actions)
            # length_reward = env.get_reward(sample_solution=actions)
            length_reward = env.get_reward(sample_solution=actions[1:])


            # rewards = tf.add(length_reward, demand_penalty)
            rewards = length_reward

            self.outputs = {'logprobs': logprobs,
                            'actions': actions,
                            'idxs': idxs,
                            'probs': probs,
                            'rewards': rewards}

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

class ActorAttentionLayer(object):
    """A generic attention module for the attention in vrp model"""
    '''This layer consider the dynamic info of the env'''

    def __init__(self, dim, name, scope='', use_tanh=False, C=10):
        self.call_time = 0
        self.use_tanh = use_tanh
        self.name = name
        self.scope = scope

        with tf.variable_scope(self.name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable(name='v_vector', shape=[1, dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v, 2, name='expand_v_vector')

        # in fact, the high level class will add prefix automatically, no matter whether it's blocked within the viriable scope
        # I write them in the variable scope just for easy to read
        self.emb_d = tf.layers.Conv1D(dim, 1, name=self.name + 'emb_d')  # conv1d
        self.emb_ld = tf.layers.Conv1D(dim, 1, name=self.name + 'emb_ld')  # conv1d_2

        self.project_d = tf.layers.Conv1D(dim, 1, name=self.name + 'proj_d')  # conv1d_1
        self.project_ld = tf.layers.Conv1D(dim, 1, name=self.name + 'proj_ld')  # conv1d_3
        self.project_query = tf.layers.Dense(dim, name=self.name + 'proj_q')  #
        self.project_ref = tf.layers.Conv1D(dim, 1, name=self.name + 'proj_ref')  # conv1d_4

        self.C = C  # tanh clip parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args:
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder.
                [batch_size x max_time x dim]

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # get the current demand and load values from environment
        self.call_time += 1
        # demand = tf.identity(env.demand,name='Actor/' + self.name + '/demand')
        demand = tf.identity(env.demand_trace[-1], name=self.name + '/demand')

        load = tf.identity(env.load_trace[-1], name=self.name + '/load')
        # load = env.load
        max_time = tf.identity(tf.shape(demand)[1], name=self.name + '/maxtime')

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = tf.identity(self.emb_d(tf.expand_dims(demand, 2)), name=self.name + '/new_emb_d')
        # d:[batch_size x max_time x dim ]
        d = tf.identity(self.project_d(emb_d), name=self.name + '/new_proj_d')

        # embed load - demand
        # emb_ld:[batch_size*beam_width x max_time x hidden_dim]
        emb_ld = tf.identity(self.emb_ld(tf.expand_dims(tf.tile(tf.expand_dims(load, 1), [1, max_time]) -
                                                        demand, 2)), name=self.name + '/new_emb_ld')
        # ld:[batch_size*beam_width x hidden_dim x max_time ]
        ld = tf.identity(self.project_ld(emb_ld), name=self.name + '/new_proj_ld')

        # expanded_q,e: [batch_size x max_time x dim]
        e = tf.identity(self.project_ref(ref), name=self.name + '/new_proj_ref')
        q = tf.identity(self.project_query(query), name=self.name + '/new_proj_q')  # [batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q, 1), [1, max_time, 1], name=self.name + '/expanded_proj_q')

        with tf.variable_scope(self.name):
            # v_view:[batch_size x dim x 1]
            v_view = tf.tile(self.v, [tf.shape(e)[0], 1, 1])

            # u : [batch_size x max_time x dim] * [batch_size x dim x 1] =
            #       [batch_size x max_time]
            u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d + ld), v_view), 2)

            if self.use_tanh:
                logits = self.C * self.tanh(u)
            else:
                logits = u

            return e, logits


class DecodeStep(object):
    def __init__(self, args, scope, GlimpseLayer, PointerLayer):
        self.hidden_dim = args['actor_hidden_dim']
        self.use_tanh = args['actor_use_tanh']
        self.tanh_exploration = args['actor_tanh_exploration']
        self.n_glimpses = args['actor_n_glimpses']
        self.mask_glimpses = args['actor_mask_glimpses']
        self.mask_pointer = args['actor_mask_pointer']

        self.scope = scope

        self.BIGNUMBER = 100000.

        # create glimpse and attention instances as well as tf.variables.
        ## create a list of class instances
        self.glimpses = [None for _ in range(self.n_glimpses)]
        for i in range(self.n_glimpses):
            self.glimpses[i] = GlimpseLayer(self.hidden_dim,
                                            use_tanh=False,
                                            name="Decoder_Attention/Glimpse_" + str(i) + "_Layer", scope=self.scope)

        # build TF variables required for pointer
        self.pointer = PointerLayer(self.hidden_dim,
                                    use_tanh=self.use_tanh,
                                    C=self.tanh_exploration,
                                    name="Decoder_Attention/Pointer_Layer", scope=self.scope)


class RNNDecodeStep(DecodeStep):
    '''
    Decodes the sequence. It keeps the decoding history in a RNN.
    '''

    def __init__(self, args, scope):
        '''
        This class does one-step of decoding which uses RNN for storing the sequence info.
        Inputs:
            ClAttention:    the class which is used for attention
            hidden_dim:     hidden dimension of RNN
            use_tanh:       whether to use tanh exploration or not
            tanh_exploration: parameter for tanh exploration
            n_glimpses:     number of glimpses
            mask_glimpses:  whether to use masking for the glimpses or not
            mask_pointer:   whether to use masking for the pointers or not
            forget_bias:    forget bias of LSTM
            rnn_layers:     number of LSTM layers
            _scope:         variable scope
