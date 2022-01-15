import scipy.io as sio
from scipy.spatial import distance_matrix
import os
import numpy as np

def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        capacity,
        seed=None,
        data_type='train'):
    '''
    This function creates VRP instances and saves them on disk. If a file is already available,
    nothing will be done.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        dir of the created data
     '''

    # set random number generator
    n_nodes = n_cust + 1
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_dir = 'vrp{}'.format(n_cust)
    fname = os.path.join(data_dir, task_dir)

    instance_type=''

    # cteate data
    if os.path.exists(fname):
        print('Data {} already exists!'.format(task_dir))
    else:
        if not os.path.isdir(fname):
            os.makedirs(fname)
        for i in range(n_problems):
            prop = np.random.random()
            if prop <= .5:
                coordinates = rnd.uniform(0, 1, size=(n_nodes, 2))
                instance_type = 'uniform'
            else:
                # coordinates = rnd.triangular(0,mode=0.5,right=1,size=(n_nodes,2))
                coordinates = np.random.normal(.5,.15,size=(n_nodes,2))
                coordinates[coordinates > 1] = 1
                coordinates[coordinates < 0] = 0
                instance_type = 'guassian'


            demand = rnd.randint(1, 10, [n_nodes, 1])
            demand[-1, :] = 0

            shortest_path_matrix = distance_matrix(coordinates, coordinates)

            task_name = 'vrp-size-{}-id-{}-{}.mat'.format(n_cust, i + 1, data_type)

            path = os.path.join(fname, task_name)

            sio.savemat(path, {'shortest_path_matrix': shortest_path_matrix, 'demand': demand,
                               'coordinates': coordinates, 'capacity': capacity, 'instance_type': instance_type})

    return fname


file_id = 0


class DataManager(object):
    def __init__(self, args, data_type):
        self.args = args
        self.path = args['data_dir'] + '/' + str(data_type)
        self.batch_size = args['batch_size']
        self.rnd = np.random.RandomState(seed=args['random_seed'])

        self.n_problems = args['instance_num']
        self.data_type = data_type
    