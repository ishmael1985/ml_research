import numpy as np
import h5py

def read_hdf5(path):
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('data'))
        label_ = np.array(hf.get('label'))
        return input_, label_



