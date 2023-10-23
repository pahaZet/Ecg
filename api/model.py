from asyncio.windows_events import NULL
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
from ECG_sequence_riginal import ECG_predict_sequence
import pandas as pd


class predict_model():
    
    @classmethod
    def boolstr_to_floatstr(self, v):
        if v >= 0.9999:
            return 1
        else:
            return 0
    
    def sexy_show(self, v):
        if v >= 0.90:
            return round(v, 4)
        else:
            return '0.0000'
    
    def predict(self, array_of_xs):
        seq = ECG_predict_sequence(array_of_xs)
        if seq.x.shape[1] != self.model.input_shape[1] or seq.x.shape[2] != self.model.input_shape[2]:
            error_text = "Входящий массив не соответствует ожиданиям модели. Он - " + str(seq.x.shape) + ". Должен быть "  + str(self.model.input_shape)
            return False, None, error_text
        y_score = self.model.predict(seq.x, verbose=0)
        return True, y_score[0], None
        

    def __init__(self, model_path = "C:\\Users\\ziruk\\YandexDisk\\ECG Project N\\Общие по ECG\\Собранные модели только с диагнозами\\Триммировано до 2700. Нормировано NK2. Нормировано от 0 до 1\\model_onlyDiaFilteredTrimmedDataset_last.hdf5"):
        self.model = load_model(model_path, compile=False) 
        self.model.compile(loss='binary_crossentropy', optimizer=Adam())
