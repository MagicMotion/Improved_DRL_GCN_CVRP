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
        dir of