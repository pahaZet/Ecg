import argparse
import numpy as np
import pandas as pd
import h5py
import pathlib
import os
import neurokit2 as nk
import matplotlib            
print(matplotlib.rcParams['backend'])
matplotlib.rcParams['backend'] = 'TkAgg' 
import matplotlib.pyplot as plt
import warnings

from sklearn import preprocessing

# Берем исходный датасет на 12 отведений.
# Оставляем только 2ое отведение.
# Вырезаем из 2ого отведения центральные 2700 значений.
# Эти значения запускаем через bypass для удаления шумов.
# Нормализуем значения от 0 до 1


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '?', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def boolstr_to_floatstr(v):
    if v == True:
        return 1
    elif v == False:
        return 0
    else:
        return v
    
def filter_and_normalize( inputdata):
    nonzeros = np.count_nonzero(inputdata)
    have_NaNs = np.isnan(inputdata).any()
    if nonzeros > 2650 and not have_NaNs:
        opty_ds = inputdata.flatten()
        try:
            #plt.clf()
            #plt.plot(inputdata)

             # фильтрация ЭКГ 
            signals, info = nk.ecg_process(opty_ds, sampling_rate=400)
            tmpx = signals['ECG_Clean'].values
            tmpx = np.reshape(tmpx, (-1, 1))
            #plt.plot(tmpx)
            
            # нормализация от 0 до 1 
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled_tmpx = min_max_scaler.fit_transform(tmpx)
            #plt.plot(scaled_tmpx)
            
            return True, scaled_tmpx
        except Exception as inst:
            print(" exc ", inst)
            return False, None
    else:
        print("nonzeros is ", nonzeros, ", have Nans - ", have_NaNs)
        return False, None
         
def makeEachEcgTrimmedFiltered(pathToFilesDir, pathToCsv):
    readedCsv = pd.read_csv(pathToCsv)
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
        
    for file in files:
        fileName = os.path.join(os.path.dirname(os.path.abspath(file)), "trimmedAndFiltered", os.path.basename(file));
        with h5py.File(fileName, 'w') as newf:
       
            f = h5py.File(file, "r")
            print("open file ", file, os.path.basename(file))
            dset = f['tracings'][:,700:3400,:] # те самые 2700 записей из центра

            csv4hdf = readedCsv.loc[readedCsv['trace_file'].isin([os.path.basename(file)])] 
            
            new_full_y = newf.create_dataset("y", (len(csv4hdf),6))
            new_full_exam_id = newf.create_dataset("exam_id", (len(csv4hdf),))
            new_full_x = newf.create_dataset("tracings", (len(csv4hdf),dset.shape[1],1))            
            
            tmpx = np.delete(dset, [0,2,3,4,5,6,7,8,9,10,11], 2)

            idx_res = 0
            einum = enumerate(f['exam_id'])
            for idx, ex in einum:
                exInCsv = readedCsv.loc[readedCsv['exam_id'] == ex]
                x_data_res, x_data = filter_and_normalize(tmpx[idx])                

                if exInCsv.size > 0 and x_data_res:
                    vals = exInCsv.values[0]
                    valsTrim = vals[4:10]
                    vectorized_res = np.vectorize(boolstr_to_floatstr)(valsTrim).astype(np.float64)
                    
                    # возьмем только с нарушениями
                    nonzeros = np.count_nonzero(vectorized_res)
                    if nonzeros > 0:
                        new_full_y[idx_res] = vectorized_res
                        print(vectorized_res, " ", idx, " ", idx_res)

                        # # фильтрация ЭКГ 
                        # signals, info = nk.ecg_process(tmpx[idx].flatten(), sampling_rate=400)
                        # filtered_x_data = signals['ECG_Clean'].values

                        # # нормализация от 0 до 1 
                        # reshaped_filtered_data = np.reshape(filtered_x_data, (-1, 1))
                        # scaled_tmpx = min_max_scaler.fit_transform(reshaped_filtered_data)
                    
                        # plt.clf()
                        # plt.plot(tmpx[idx_res])
                        # plt.plot(filtered_x_data)
                        # plt.plot(scaled_tmpx)

                        new_full_x[idx_res] = x_data
                        new_full_exam_id[idx_res] = ex
                        
                        # plt.clf()
                        # plt.plot(x_data)
                    
                        idx_res += 1
                else:
                    print("FOR ex #", ex," ",idx," no data in CSV!!!")
            
            newf.create_dataset("actual_data_count", idx_res)
            print("file ", file, " complite process, actual_data_count", idx_res)   
            f.close
            break


# выборка всех экг с диагнозами из
# уже оттримированных датасетов
def make_each_ecg_trimmed_filtered_only_dia_from_all_already_trimmed_datasets(pathToFilesDir, path_to_result_hdf5):
    
    files = [f for f in pathlib.Path(pathToFilesDir).glob("*.hdf5")]
    fileName = path_to_result_hdf5;
    new_x = []
    new_y = []
    idx_res = 0
    
    with h5py.File(fileName, 'w') as newf:    
        for file in files:
            f = h5py.File(file, "r")
            print("open file ", file, os.path.basename(file))
            tmpx = f['tracings']
            tmpy = f['y']
            einum = enumerate(tmpx)
            for idx, t in einum:
                # возьмем только с нарушениями
                nonzeros = np.count_nonzero(tmpy[idx])
                if nonzeros > 0:
                    x_data_res, x_data = filter_and_normalize(t)                
                    if x_data_res:
                        new_x.append(x_data)
                        new_y.append(tmpy[idx])

                        # plt.clf()
                        # plt.plot(x_data)
                    
                        print(tmpy[idx],", idx_res - ", idx_res)

                        idx_res += 1
                    else:
                        print("no exec filter_and_normalize, idx - ",idx)
            print("file ", file, " complite process, actual_data_count", idx_res)   
            f.close
        
        new_full_y = newf.create_dataset("y", np.array(new_y))
        new_full_x = newf.create_dataset("tracings", np.array(new_x))   
        newf.create_dataset("actual_data_count", idx_res)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--path_to_hdf5', type=str, default='C:\\paha\\ECG\\Code15data\\exams\\oneDimTrimmed', help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_result_hdf5', type=str, default='C:\\paha\\ECG\\Code15data\\exams\\oneDimTrimmed\\only_dia.hdf5', help='path to result hdf5 file containing DIA tracings')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    

    #makeEachEcgTrimmedFiltered("C:\\paha\\ECG\\Code15data\\exams\\","C:\\paha\\ECG\\Code15data\\exams.csv")
    make_each_ecg_trimmed_filtered_only_dia_from_all_already_trimmed_datasets(args.path_to_hdf5, path_to_result_hdf5)
