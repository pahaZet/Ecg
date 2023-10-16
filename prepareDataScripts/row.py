from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,CSVLogger, EarlyStopping)
from keras.models import Sequential
from model import get_model
import argparse
from datasets import ECGSequence
import numpy as np
import pandas as pd
import h5py
import pathlib
import os
import neurokit2 as nk


from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from keras.models import Model


def boolstr_to_floatstr(v):
    if v == True:
        return 1
    elif v == False:
        return 0
    else:
        return v

def makeOneEcgDataset(pathToFilesDir, pathToCsv):
    # new hdf5 full dataset
    readedCsv = pd.read_csv(pathToCsv)
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
    fileName = os.path.join(pathToFilesDir, "new1dim", "all4.hdf5");
    fileidx=0
    with h5py.File(fileName, 'w') as newf:
        
        new_full_y = newf.create_dataset("y", (80000,6))
        new_full_exam_id = newf.create_dataset("exam_id", (80000,))
        new_full_x = newf.create_dataset("tracings", (80000,4096,1))          
        # new_full_y = newf.create_dataset("y", (len(readedCsv),6))
        # new_full_exam_id = newf.create_dataset("exam_id", (len(readedCsv),))
        # new_full_x = newf.create_dataset("tracings", (len(readedCsv),4096,1))  
        idx = 0
        
        for file in files:
            if fileidx >= 4:
                break
            
            f = h5py.File(file, "r")
            print("open file ", file, os.path.basename(file))
            print(f.keys())
            dset = f['tracings']
            print(dset.shape)
            print(len(dset))
            
            csv4hdf = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(file)])] 

            tmpx = np.delete(dset, [0,2,3,4,5,6,7,8,9,10,11], 2)

            
            einum = enumerate(f['exam_id'])
            for iii, ex in einum:
                exInCsv = csv4hdf.loc[readedCsv['exam_id'] == ex]
                if exInCsv.size > 0:
                    vals = exInCsv.values[0]
                    valsTrim = vals[4:10]
                    vectorized_res = np.vectorize(boolstr_to_floatstr)(valsTrim).astype(np.float64)
                    
                    new_full_y[idx] = vectorized_res
                    new_full_x[idx] = tmpx[iii]
                    new_full_exam_id[idx] = ex
                    idx+=1
                else:
                    print("FOR ex #", ex," no data in CSV!!!")
                    
            print("file ", file, " complite process")   
            f.close
            fileidx+=1
    
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def makeOneEcgDatasetWithCurrentDia(pathToFilesDir, pathToCsv, findPattern):
    # new hdf5 full dataset
    readedCsv = pd.read_csv(pathToCsv)

    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
    total_foundBadItems = 0
    total_foundNormalItems = 0
    
    new_x_dia = np.zeros((3000, 2700, 1))
    new_x_norm = np.zeros((3000, 2700, 1))

    for file in files:
        f = h5py.File(file, "r")
        print("open file ", file, os.path.basename(file))
        print(f.keys())
        dset = f['tracings'][:,700:3400,:]
        csv4hdf = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(file)])] 
        tmpx = np.delete(dset, [0,2,3,4,5,6,7,8,9,10,11], 2)



        einum = enumerate(f['exam_id'])
        for idx, ex in einum:
            exInCsv = readedCsv.loc[readedCsv['exam_id'] == ex]
            if exInCsv.size > 0:
                vals = exInCsv.values[0]
                valsTrim = vals[4:10]
                vectorized_res = np.vectorize(boolstr_to_floatstr)(valsTrim).astype(int)
                if np.array_equal(vectorized_res, findPattern):
                    if total_foundBadItems < new_x_dia.shape[0]:
                        dia = tmpx[idx]
                        #normalized_dia = NormalizeData(dia)
                        new_x_dia[total_foundBadItems] = dia
                        total_foundBadItems+=1
                    
                elif np.array_equal(vectorized_res, np.array([0,0,0,0,0,0])):
                    if total_foundNormalItems < new_x_norm.shape[0]:
                        nrm = tmpx[idx]
                        #normalized_nrm = NormalizeData(nrm)
                        new_x_norm[total_foundNormalItems] = nrm
                        total_foundNormalItems+=1
            else:
                print("FOR ex #", ex," no data in CSV!!!")
            
            if total_foundBadItems >= new_x_dia.shape[0] and total_foundNormalItems >= new_x_norm.shape[0]:
                break
                
        print("if file ", file, " bad items - ", total_foundBadItems, ", normal ECGs - ", total_foundNormalItems)   
        f.close
        if total_foundBadItems >= new_x_dia.shape[0] and total_foundNormalItems >= new_x_norm.shape[0]:
            break

    fileName = os.path.join(os.path.dirname(os.path.abspath(file)), "oneDimTrimmedOnlyNormAnd_ST", os.path.basename(file));
    with h5py.File(fileName, 'w') as newf:
        new_full_y = newf.create_dataset("y", (len(new_x_dia)*2,6))
        new_full_x = newf.create_dataset("tracings", (len(new_x_dia)*2,dset.shape[1],1))
        
        normal_array = np.array([0,0,0,0,0,0])
        dia_array = findPattern

        odd = False
        dia_idx = 0
        norm_idx = 0
        for i in range(0, len(new_x_dia)*2):
            if odd:
                new_full_x[i] = new_x_dia[dia_idx]
                new_full_y[i] = dia_array
                dia_idx+=1
                odd = not odd
            else:
                new_full_x[i] = new_x_norm[norm_idx]
                new_full_y[i] = normal_array
                norm_idx+=1
                odd = not odd
                
    print("all done. last element val = ", new_full_x[len(new_x_dia)*2 - 1])       

         
def makeEachEcgTrimmed(pathToFilesDir, pathToCsv):
    # new hdf5 full dataset
    readedCsv = pd.read_csv(pathToCsv)
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
        
    for file in files:
        fileName = os.path.join(os.path.dirname(os.path.abspath(file)), "oneDimTrimmed", os.path.basename(file));
        with h5py.File(fileName, 'w') as newf:
       
            f = h5py.File(file, "r")
            print("open file ", file, os.path.basename(file))
            dset = f['tracings'][:,700:3400,:]

            csv4hdf = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(file)])] 
            
            new_full_y = newf.create_dataset("y", (len(csv4hdf),6))
            new_full_exam_id = newf.create_dataset("exam_id", (len(csv4hdf),))
            new_full_x = newf.create_dataset("tracings", (len(csv4hdf),dset.shape[1],1))            
            
            tmpx = np.delete(dset, [0,2,3,4,5,6,7,8,9,10,11], 2)


            idx = 0
            einum = enumerate(f['exam_id'])
            for idx, ex in einum:
                exInCsv = readedCsv.loc[readedCsv['exam_id'] == ex]
                if exInCsv.size > 0:
                    vals = exInCsv.values[0]
                    valsTrim = vals[4:10]
                    vectorized_res = np.vectorize(boolstr_to_floatstr)(valsTrim).astype(np.float64)
                    
                    new_full_y[idx] = vectorized_res
                    new_full_x[idx] = tmpx[idx]
                    new_full_exam_id[idx] = ex
                else:
                    print("FOR ex #", ex," ",idx," no data in CSV!!!")
                    
            print("file ", file, " complite process")   
            f.close
 
            
def makeEachEcgTrimmedFiltered(pathToFilesDir, pathToCsv):
    # new hdf5 full dataset
    readedCsv = pd.read_csv(pathToCsv)
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
        
    for file in files:
        fileName = os.path.join(os.path.dirname(os.path.abspath(file)), "trimmedAndFiltered", os.path.basename(file));
        with h5py.File(fileName, 'w') as newf:
       
            f = h5py.File(file, "r")
            print("open file ", file, os.path.basename(file))
            dset = f['tracings'][:,700:3400,:]

            csv4hdf = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(file)])] 
            
            new_full_y = newf.create_dataset("y", (len(csv4hdf),6))
            new_full_exam_id = newf.create_dataset("exam_id", (len(csv4hdf),))
            new_full_x = newf.create_dataset("tracings", (len(csv4hdf),dset.shape[1],1))            
            
            tmpx = np.delete(dset, [0,2,3,4,5,6,7,8,9,10,11], 2)
            opty_ds = tmpx[0].flatten()
            signals, info = nk.ecg_process(opty_ds, sampling_rate=400)
            
            tmpx = signals['ECG_Clean'].values
            tmpx = np.reshape(tmpx, (-1, 1))

            idx = 0
            einum = enumerate(f['exam_id'])
            for idx, ex in einum:
                exInCsv = readedCsv.loc[readedCsv['exam_id'] == ex]
                if exInCsv.size > 0:
                    vals = exInCsv.values[0]
                    valsTrim = vals[4:10]
                    vectorized_res = np.vectorize(boolstr_to_floatstr)(valsTrim).astype(np.float64)
                    
                    new_full_y[idx] = vectorized_res
                    new_full_x[idx] = tmpx[idx]
                    new_full_exam_id[idx] = ex
                else:
                    print("FOR ex #", ex," ",idx," no data in CSV!!!")
                    
            print("file ", file, " complite process")   
            f.close
            break
            
def makeEachEcgDataset(pathToFilesDir, pathToCsv):
    # new hdf5 full dataset
    readedCsv = pd.read_csv(pathToCsv)
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
        
    for file in files:
        fileName = os.path.join(os.path.dirname(os.path.abspath(file)), "new1dim", os.path.basename(file));
        with h5py.File(fileName, 'w') as newf:
       
            f = h5py.File(file, "r")
            print("open file ", file, os.path.basename(file))
            print(f.keys())
            dset = f['tracings']
            print(dset.shape)
            print(len(dset))
            
            csv4hdf = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(file)])] 
            
            new_full_y = newf.create_dataset("y", (len(csv4hdf),6))
            new_full_exam_id = newf.create_dataset("exam_id", (len(csv4hdf),))
            new_full_x = newf.create_dataset("tracings", (len(csv4hdf),4096,1))            
            
            tmpx = np.delete(dset, [0,2,3,4,5,6,7,8,9,10,11], 2)

            #new_full_exam_id = f['exam_id']

            idx = 0
            einum = enumerate(f['exam_id'])
            for idx, ex in einum:
                exInCsv = readedCsv.loc[readedCsv['exam_id'] == ex]
                if exInCsv.size > 0:
                    vals = exInCsv.values[0]
                    valsTrim = vals[4:10]
                    vectorized_res = np.vectorize(boolstr_to_floatstr)(valsTrim).astype(np.float64)
                    
                    new_full_y[idx] = vectorized_res
                    new_full_x[idx] = tmpx[idx]
                    new_full_exam_id[idx] = ex
                else:
                    print("FOR ex #", ex," no data in CSV!!!")
                    
            print("file ", file, " complite process")   
            f.close
        

def finfZerosInX(pathToFilesDir):
    # new hdf5 full dataset
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
        
    for file in files:
        f = h5py.File(file, "r")
        print("open file ", file, os.path.basename(file))
        dset = f['tracings']

        einum0 = enumerate(dset)
        for idx0, ex0 in einum0:
            print("record ", idx0)
            einum1 = enumerate(ex0)
            foundArray = []
            for idx1, ex1 in einum1:
                if ex1[0] == 0:
                    foundArray.append(idx1)
                idx1+=1
            if len(foundArray) > 0:
                print(foundArray)
                    
        print("file ", file, " complite process")   
        f.close
                    
if __name__ == "__main__":
    dia = np.array([0,0,0,0,0,1]) # sinus tachycardia (ST)
    #dia = np.array[0,0,0,1,0,0]) # sinus bradycardia (SB)
    #dia = np.array[0,0,0,0,1,0]) # atrial fibrillation (AF)
    # dia = [0,0,0,0,0,1] # sinus tachycardia (ST)
    # dia = [0,0,0,0,0,1] # sinus tachycardia (ST)
    # dia = [0,0,0,0,0,1] # sinus tachycardia (ST)
    
    #makeOneEcgDatasetWithCurrentDia("C:\\paha\\ECG\\Code15data\\exams\\","C:\\paha\\ECG\\Code15data\\exams.csv", dia) 
    makeEachEcgTrimmedFiltered("C:\\paha\\ECG\\Code15data\\exams\\","C:\\paha\\ECG\\Code15data\\exams.csv")
    #finfZerosInX("C:\\paha\\ECG\\Code15data\\exams\\new1dim")


    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')

    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()
    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        "C:\\paha\\ECG\\Code15data\\exams\\exams_part0.hdf5", args.dataset_name, "C:\\paha\\ECG\\Code15data\\exams.csv", batch_size, args.val_split)

    verbose, epochs, batch_size = 1, 10, 32
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(4096,1)))
    model.add(Conv1D(filters=196, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss=loss, optimizer=opt)

	# fit network
    model.fit(train_seq, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    