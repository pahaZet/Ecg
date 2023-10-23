import h5py
import math
import pandas as pd
from keras.utils import Sequence
import numpy as np
import ast
import os
import neurokit2 as nk
from numpy import newaxis
from sklearn import preprocessing

class ECG_predict_sequence(Sequence):
    @classmethod
   
    def boolstr_to_floatstr(self,v):
        if v == True:
            return 1
        elif v == False:
            return 0
        else:
            return v
        
    # фильтр от шума и нормализаци¤ от 0 до 1
    # уже оттримированных датасетов
    def filter_and_normalize(self, inputdata):
        nonzeros = np.count_nonzero(inputdata)
        have_NaNs = np.isnan(inputdata).any()
        if nonzeros > 2650 and not have_NaNs:
            opty_ds = inputdata.flatten()
            try:
                 # фильтраци¤ Ё √ 
                # signals, info = nk.ecg_process(opty_ds, sampling_rate=400)
                # tmpx = signals['ECG_Clean'].values
                # tmpx = np.reshape(tmpx, (-1, 1))
                # #plt.plot(tmpx)
            
                # # нормализаци¤ от 0 до 1 
                # min_max_scaler = preprocessing.MinMaxScaler()
                # scaled_tmpx = min_max_scaler.fit_transform(tmpx)
            
                return True, inputdata #scaled_tmpx
            except Exception as inst:
                return False, None
        else:
            #print("nonzeros is ", nonzeros, ", have Nans - ", have_NaNs)
            return False, None
 

    def __init__(self, array):

        self.x = np.array(array)
        self.x = self.x[:, :, newaxis]
        for idx, x in enumerate(self.x):
            res, filtered_x = self.filter_and_normalize(x)
            if res:
                self.x[idx] = filtered_x
            else:
                print("error filter data idx ", idx)
        self.batch_size = 8
        self.start_idx = 0
        self.idx = 0
        self.end_idx = len(self.x)
        

    @property

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        return np.array(self.x[start:end, :, :])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

