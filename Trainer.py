
from Model.model import *
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class Trainer(object):
    def __init__(self, args, model, env):
        with tf.variable_scope('Trainer'):
            self.args = args
            self.model = model
            self.environment = env

            self.datamanager = DataManager(self.args, 'train')
            self.datamanager.create_data()

            self.total_epoch = args['trainer_epoch']

            self.output = model.outputs
            self.estimate = model.critic.reward_predict
            self.reward = self.output['rewards']
            self.logprobs = self.output['logprobs']
            self.probs = self.output['probs']
            self.idxs = self.output['idxs']
            self.actions = self.output['actions']

            self.frozen_estimate = tf.stop_gradient(self.estimate)
            self.frozen_reward = tf.stop_gradient(self.reward)

            self.actor_loss = tf.reduce_mean(
                tf.multiply((self.frozen_estimate - self.frozen_reward), tf.add_n(self.logprobs)), 0)
            tf.summary.scalar('Loss/Actor Loss', self.actor_loss)

            self.critic_loss = tf.losses.mean_squared_error(self.frozen_reward, self.estimate)
            tf.summary.scalar('Loss/Critic Loss', self.critic_loss)

            if args['embedding_type'] == 'gcn':
                self.gcn_loss = self.actor_loss + self.critic_loss
                tf.summary.scalar('Loss/GCN Loss', self.gcn_loss)
            elif args['embedding_type'] == 'linear_embedding':
                self.linear_embedding_loss = self.actor_loss + self.critic_loss

            # optimizers
            self.actor_optim = tf.train.AdamOptimizer(args['actor_lr'])
            self.critic_optim = tf.train.AdamOptimizer(args['critic_lr'])

            # compute gradients
            self.actor_gra_and_var = self.actor_optim.compute_gradients(self.actor_loss,
                                                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                          scope='Model/Actor'))

            self.critic_gra_and_var = self.critic_optim.compute_gradients(self.critic_loss,
                                                                          tf.get_collection(
                                                                              tf.GraphKeys.GLOBAL_VARIABLES,
                                                                              scope='Model/Critic'))

            # clip gradients
            self.clip_actor_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                                           for grad, var in self.actor_gra_and_var]
            self.clip_critic_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                                            for grad, var in self.critic_gra_and_var]

            # apply gradients
            # apply gradient descent by default
            self.actor_train_step = self.actor_optim.apply_gradients(self.clip_actor_gra_and_var)
            self.critic_train_step = self.critic_optim.apply_gradients(self.clip_critic_gra_and_var)

            if args['embedding_type'] == 'gcn':
                self.gcn_optim = tf.train.AdamOptimizer(args['gcn_lr'])
                self.gcn_gra_and_var = self.gcn_optim.compute_gradients(self.gcn_loss,
                                                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                          scope='Model/GCN'))

                self.clip_gcn_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                                             for grad, var in self.gcn_gra_and_var]

                self.gcn_train_step = self.gcn_optim.apply_gradients(self.clip_gcn_gra_and_var)

                self.train_step = [self.actor_train_step,
                                   self.critic_train_step,
                                   self.gcn_train_step]

            elif args['embedding_type'] == 'linear_embedding':