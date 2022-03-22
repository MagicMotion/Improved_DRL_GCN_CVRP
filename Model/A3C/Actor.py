
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