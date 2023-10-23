import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,CSVLogger, EarlyStopping)
import argparse 
import numpy as np
import tensorflow as tf
import uuid
import keras.backend as backend

import h5py
import math
import pandas as pd
from keras.utils import Sequence

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
    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):


        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]

        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.y = self.f['y']
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

class ResidualUnit(object):
    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]

def get_model(n_classes, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(2700, 1), dtype=np.float32, name='signal')
    x = signal
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x = Flatten()(x)
    diagn = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn)
    return model

if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print("Num GPUs used: ", len(gpus))
    print("Num CPUs used: ", len(cpus))
    print(tf.sysconfig.get_build_info())

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    val_split = 0.02
    # Optimization setting.æs
    loss = 'binary_crossentropy'
    #loss = 'categorical_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                               patience=9, verbose=1, mode='auto')
    

    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 10, verbose=1),
                 #early_stop]                  
                 EarlyStopping(patience=9,  # patience should be larger than the one in reducelronplateau
                               min_delta=0.00001, verbose=1)]

    train_seq, valid_seq = ECGSequence.get_train_and_val(
        "C:\\paha\\ECG\\Code15data\\onlyDiaFilteredTrimmed_notNoralized_0_1_not_filtered_Dataset.hdf5", "tracings", None, batch_size, val_split)

    # If you are continuing an interrupted section, uncomment line bellow:
    # model = tf.keras.models.load_model("C:\\paha\\ECG\\Code15data\\123\\backup_model_last.hdf5", compile=False)

    #model.summary()
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    # Create log

    callbacks += [TensorBoard(log_dir="C:\\paha\\ECG\\Code15data\\123\\Logs", write_graph=False),
                  CSVLogger("C:\\paha\\ECG\\Code15data\\123\\trained.log", append=True)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint("C:\\paha\\ECG\\Code15data\\123\\backup_model_last.hdf5"),
                  ModelCheckpoint("C:\\paha\\ECG\\Code15data\\123\\backup_model_best.hdf5", save_best_only=True, monitor='val_loss')]


    # Train neural network
    history = model.fit(train_seq,
                    epochs=50,
                    batch_size=batch_size,
                    #initial_epoch=13,  # If you are continuing a interrupted section change here
                    callbacks=callbacks,
                    validation_data=valid_seq,
                    verbose=1,
                    )




