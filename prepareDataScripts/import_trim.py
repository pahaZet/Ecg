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


_file_name = ""

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
                    new_x.append(t)
                    new_y.append(tmpy[idx])
                    idx_res += 1
                    printProgressBar(idx + 1, len(tmpx), prefix = os.path.basename(file), suffix = 'Complete', length = 50)

            print("file ", file, " complite process, actual_data_count", idx_res)   
            f.close

        
        new_full_y = newf.create_dataset("y", data=new_y)
        new_full_x = newf.create_dataset("tracings", data=new_x)   

    print("done...")   




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--path_to_hdf5', type=str, default='C:\\paha\\ECG\\Code15data\\exams\\oneDimTrimmed', help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_result_hdf5', type=str, default='C:\\paha\\ECG\\Code15data\\exams\\oneDimTrimmed\\only_dia_default.hdf5', help='path to result hdf5 file containing DIA tracings')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    

    #makeEachEcgTrimmedFiltered("C:\\paha\\ECG\\Code15data\\exams\\","C:\\paha\\ECG\\Code15data\\exams.csv")
    make_each_ecg_trimmed_filtered_only_dia_from_all_already_trimmed_datasets(args.path_to_hdf5, args.path_to_result_hdf5)
