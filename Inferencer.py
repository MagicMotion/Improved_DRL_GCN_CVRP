
from Model.model import *


class Inferencer(object):
    def __init__(self, model_path, model, env):
        self.model = model
        self.environment = env

        self.output = model.outputs

        self.model_path = model_path
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt:
            print('Loading model...\npath: ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('Fatal Error,can\'t find trained model!')

    def watch_variables(self):
        # Variables are independent with inputs
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print('Model variables watch:')
        for item in var_list:
            print(item.name, ' shape:', item.shape)
            print(self.sess.run(item))

    def __call__(self, input_data):
        # self.watch_variables()
        self.environment.info_inspect(input_data, self.model, self.sess)
        self.environment.get_routes(input_data, self.model, self.sess)


if __name__ == '__main__':
    args = {}

    # Trainer
    args['trainer_save_interval'] = 10000
    args['trainer_inspect_interval'] = 10000
    args['trainer_model_dir'] = 'model_trained/'
    args['trainer_epoch'] = 260000
    args['batch_size'] = 1
    args['keep_prob'] = 1

    # Environment
    args['n_customers'] = 10
    args['data_dir'] = 'data/'
    args['random_seed'] = 1
    args['instance_num'] = 5000
    args['capacity'] = 20

    # Network
    # embedding type
    # {'gcn','linear_embedding'}
    args['embedding_type'] = 'linear_embedding'

    #linear embedding
    args['linear_embedding_num_units'] = 128
    args['linear_embedding_lr'] = 0.01

    # GCN
    args['gcn_lr'] = 0.01