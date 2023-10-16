import h5py
import math
import pandas as pd
from keras.utils import Sequence
import numpy as np
import ast
import os


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        # readedCsv = pd.read_csv(path_to_csv)
        # readedCsv = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(path_to_hdf5)])] 
        file2len = h5py.File(path_to_hdf5, "r")
        n_samples = len(file2len['y'])
        file2len.close
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        
        print(train_seq.x.shape)
        print(train_seq.y.shape)
        print(valid_seq.x.shape)
        print(valid_seq.y.shape)
        

        return train_seq, valid_seq

    def boolstr_to_floatstr(self,v):
        if v == True:
            return 1
        elif v == False:
            return 0
        else:
            return v

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):


        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        #self.x = np.delete(self.f[hdf5_dset], [0,2,3,4,5,6,7,8,9,10,11], 2)


        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.y = self.f['y']
        
        # if path_to_csv is None:
        #     self.y = None
        # else:
        #     readedCsv = pd.read_csv(path_to_csv)
        #     readedCsv = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(path_to_hdf5)])] 
        #     self.y = np.vectorize(self.boolstr_to_floatstr)(readedCsv.values[:,4:10]).astype(float)
            




    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
