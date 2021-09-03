import glob
import tensorflow as tf
import pickle5 as pickle
def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
load_pkl('network_final-poisson-n2n.pickle')
