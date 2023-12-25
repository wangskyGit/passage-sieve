import pickle
import numpy as np
import os 
from tqdm import tqdm


def concat(arrays):
    return np.concatenate(arrays, axis=0)

def read_file(file_list):
    arrays = []
    for file in tqdm(file_list):
        pickle_path = os.path.join('temp', file)
        with open(pickle_path, 'rb') as handle:
            b = pickle.load(handle)
            arrays.append(b)
    return concat(arrays)

def save_file(arrays, pickle_path):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(arrays, handle, protocol=4)

def delete_file(file_list):
    for file in file_list:
        path = os.path.join('temp', file)
        os.remove(path)
 
i = 1
psg_embed_file_list = ["{1}_data_obj_{0}_{2}.pb".format(str(i), 'psg_embed_id', j)
                        for j in range(8)]

# data_arrays = read_file(psg_embed_file_list)
# new_pickle_path = os.path.join('temp', "{1}_data_obj_{0}.pb".format(i, 'psg_embed_id'))
# save_file(data_arrays, new_pickle_path)

delete_file(psg_embed_file_list)


